import json
import os
import ssl
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
HISTORY_FILE = ROOT / "conversation_history.json"
MODEL_OPTIONS = {
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "model_env": "LLM_MODEL",
        "model_default": "deepseek-v4-pro",
        "base_url_env": "LLM_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
        "api_key_env": "LLM_API_KEY",
        "disable_reasoning_for_temperature": True,
    },
    "gemma-3-4b": {
        "label": "Gemma 3 4B",
        "model_env": "HF_GEMMA_MODEL",
        "model_default": "google/gemma-3-4b-it:cheapest",
        "base_url_env": "HF_BASE_URL",
        "base_url_default": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "disable_reasoning_for_temperature": False,
        "pricing": {"input": 0.05, "output": 0.10},
    },
    "qwen3-8b": {
        "label": "Qwen 3 8B",
        "model_env": "HF_QWEN_MODEL",
        "model_default": "Qwen/Qwen3-8B:cheapest",
        "base_url_env": "HF_BASE_URL",
        "base_url_default": "https://router.huggingface.co/v1",
        "api_key_env": "HF_TOKEN",
        "disable_reasoning_for_temperature": False,
        "pricing": {"input": 0.07, "output": 0.18},
    },
}


class AgentConfigurationError(RuntimeError):
    """Raised when an agent cannot start because its API key is missing."""


def load_local_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


load_local_env()


class ConversationStore:
    """Thread-safe JSON storage for conversations that survive restarts."""

    _lock = threading.RLock()

    def __init__(self, path=HISTORY_FILE, max_messages=20):
        self.path = Path(path)
        self.max_messages = max_messages

    def get(self, session_id):
        with self._lock:
            entry = self._read_all().get(session_id, [])
            messages = entry.get("messages", []) if isinstance(entry, dict) else entry
            return [dict(message) for message in messages]

    def list(self):
        with self._lock:
            items = []
            for session_id, entry in self._read_all().items():
                messages = entry.get("messages", []) if isinstance(entry, dict) else entry
                if not messages:
                    continue
                first_user_message = next(
                    (
                        message.get("content", "")
                        for message in messages
                        if message.get("role") == "user"
                    ),
                    "Новый диалог",
                )
                title = " ".join(first_user_message.split())[:48]
                items.append(
                    {
                        "id": session_id,
                        "title": title + ("…" if len(first_user_message) > 48 else ""),
                        "message_count": len(messages),
                        "updated_at": entry.get("updated_at", "")
                        if isinstance(entry, dict)
                        else "",
                    }
                )
            return sorted(items, key=lambda item: item["updated_at"], reverse=True)

    def save(self, session_id, messages):
        clean_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages[-self.max_messages :]
        ]
        with self._lock:
            conversations = self._read_all()
            conversations[session_id] = {
                "messages": clean_messages,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_all(conversations)

    def clear(self, session_id):
        with self._lock:
            conversations = self._read_all()
            if session_id in conversations:
                del conversations[session_id]
                self._write_all(conversations)

    def _read_all(self):
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write_all(self, conversations):
        temporary_path = self.path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(conversations, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.path)


class SimpleAgent:
    """Encapsulates one LLM model, its HTTP request, and response parsing."""

    def __init__(
        self,
        model_key="deepseek-v4-pro",
        api_key=None,
        session_id=None,
        history_file=HISTORY_FILE,
    ):
        if model_key not in MODEL_OPTIONS:
            raise ValueError("Неизвестная модель агента.")
        self.model_key = model_key
        self.config = MODEL_OPTIONS[model_key]
        self.api_key = api_key or os.environ.get(self.config["api_key_env"])
        self.session_id = session_id
        self.store = ConversationStore(history_file) if session_id else None
        if not self.api_key:
            raise AgentConfigurationError(
                f"На сервере не настроен {self.config['api_key_env']} "
                f"для модели {self.config['label']}."
            )

    @property
    def label(self):
        return self.config["label"]

    def ask(self, user_request, instructions=None, temperature=None):
        """Accept a user request and return only the model's text answer."""
        return self.run(user_request, instructions, temperature)["text"]

    @property
    def history(self):
        return self.store.get(self.session_id) if self.store else []

    def chat(self, user_request, instructions=None, temperature=None):
        """Continue a persistent conversation and save the new exchange."""
        if not self.store:
            raise AgentConfigurationError("Для диалога не указан session_id.")
        messages = self.history
        messages.append({"role": "user", "content": user_request})
        result = self.run(messages, instructions, temperature)
        messages.append({"role": "assistant", "content": result["text"]})
        self.store.save(self.session_id, messages)
        return result

    def clear_history(self):
        if self.store:
            self.store.clear(self.session_id)

    def replace_last_answer(self, answer):
        """Keep stored history identical to the answer shown in the interface."""
        if not self.store:
            return
        messages = self.history
        if messages and messages[-1].get("role") == "assistant":
            messages[-1]["content"] = answer
            self.store.save(self.session_id, messages)

    def run(self, user_request, instructions=None, temperature=None):
        """Send a request to the LLM and return text plus execution metrics."""
        request_body = self._build_request_body(
            user_request, instructions, temperature
        )
        result, elapsed_seconds = self._send(request_body)
        usage = result.get("usage") or {}
        cost_usd, pricing_note = self._estimate_cost(usage)
        return {
            "text": self._extract_text(result),
            "elapsed_seconds": round(elapsed_seconds, 3),
            "usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            "cost_usd": round(cost_usd, 8),
            "pricing_note": pricing_note,
        }

    def _build_request_body(self, user_request, instructions, temperature):
        request_body = {
            "model": os.environ.get(
                self.config["model_env"], self.config["model_default"]
            ),
            "input": user_request,
        }
        if instructions:
            request_body["instructions"] = instructions
        if temperature is not None:
            request_body["temperature"] = temperature
            if self.config["disable_reasoning_for_temperature"]:
                request_body["reasoning"] = {"effort": "none"}
        return request_body

    def _send(self, request_body):
        api_url = os.environ.get(
            self.config["base_url_env"], self.config["base_url_default"]
        ).rstrip("/")
        request = Request(
            f"{api_url}/responses",
            data=json.dumps(request_body).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        ssl_context = ssl.create_default_context(
            cafile=str(SYSTEM_CA_FILE) if SYSTEM_CA_FILE.exists() else None
        )
        started_at = time.perf_counter()
        for attempt in range(2):
            try:
                with urlopen(request, timeout=90, context=ssl_context) as response:
                    result = json.load(response)
                return result, time.perf_counter() - started_at
            except HTTPError as error:
                if attempt == 0 and error.code in {429, 500, 502, 503, 504}:
                    continue
                raise

    @staticmethod
    def _extract_text(result):
        return "".join(
            content.get("text", "")
            for item in result.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )

    def _estimate_cost(self, usage):
        input_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cached_tokens = int(
            (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        )
        if self.model_key == "deepseek-v4-pro":
            now = datetime.now(timezone.utc)
            is_peak = now.weekday() < 5 and (
                1 <= now.hour < 4 or 6 <= now.hour < 10
            )
            multiplier = 2 if is_peak else 1
            cost = (
                cached_tokens * 0.022 * multiplier
                + max(0, input_tokens - cached_tokens) * 0.66 * multiplier
                + output_tokens * 1.98 * multiplier
            ) / 1_000_000
            period = "peak" if is_peak else "off-peak"
            return cost, f"Расчёт по тарифу DeepSeek {period}"

        pricing = self.config["pricing"]
        cost = (
            input_tokens * pricing["input"] + output_tokens * pricing["output"]
        ) / 1_000_000
        return cost, "Расчёт по тарифу Hugging Face :cheapest до бесплатных кредитов"
