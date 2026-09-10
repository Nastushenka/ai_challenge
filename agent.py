import json
import math
import os
import sqlite3
import ssl
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
HISTORY_DATABASE = ROOT / "conversation_history.db"
LEGACY_HISTORY_FILE = ROOT / "conversation_history.json"
MODEL_OPTIONS = {
    "deepseek-v4-pro": {
        "label": "DeepSeek V4 Pro",
        "model_env": "LLM_MODEL",
        "model_default": "deepseek-v4-pro",
        "base_url_env": "LLM_BASE_URL",
        "base_url_default": "https://api.deepseek.com",
        "api_key_env": "LLM_API_KEY",
        "disable_reasoning_for_temperature": True,
        "context_window": 1_000_000,
        "context_env": "DEEPSEEK_CONTEXT_WINDOW",
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
        "context_window": 131_072,
        "context_env": "GEMMA_CONTEXT_WINDOW",
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
        "context_window": 32_768,
        "context_env": "QWEN_CONTEXT_WINDOW",
    },
}


class AgentConfigurationError(RuntimeError):
    """Raised when an agent cannot start because its API key is missing."""


class ContextWindowExceeded(RuntimeError):
    """Raised before an API call when the prepared context cannot fit."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            "Контекст переполнен: "
            f"нужно примерно {report['estimated_input_tokens']:,} токенов, "
            f"лимит модели — {report['context_limit']:,}."
        )


def estimate_text_tokens(text):
    """Estimate tokens without a model-specific tokenizer (good for preflight/UI)."""
    text = str(text or "")
    if not text:
        return 0
    # Russian text commonly uses about three characters per token.  This is an
    # intentionally conservative estimate; the API usage remains the exact source.
    return max(1, math.ceil(len(text) / 3))


def estimate_messages_tokens(messages):
    """Estimate message content plus the small role/message framing overhead."""
    return sum(estimate_text_tokens(message.get("content", "")) + 4 for message in messages)


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
    """Thread-safe SQLite storage for complete persistent conversations."""

    _lock = threading.RLock()

    def __init__(self, path=HISTORY_DATABASE, legacy_json_path=None):
        self.path = Path(path)
        self.legacy_json_path = (
            Path(legacy_json_path)
            if legacy_json_path
            else LEGACY_HISTORY_FILE if self.path == HISTORY_DATABASE else None
        )
        self._initialize()
        self._migrate_legacy_json()

    def get(self, session_id):
        with self._lock:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT role, content FROM messages "
                    "WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def list(self):
        with self._lock:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT conversations.session_id, conversations.title, "
                    "COUNT(messages.id), conversations.updated_at "
                    "FROM conversations "
                    "LEFT JOIN messages ON messages.session_id = conversations.session_id "
                    "GROUP BY conversations.session_id "
                    "ORDER BY conversations.updated_at DESC"
                ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "message_count": row[2],
                "updated_at": row[3],
            }
            for row in rows
        ]

    def save(self, session_id, messages):
        clean_messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        if not clean_messages:
            self.clear(session_id)
            return
        now = datetime.now(timezone.utc).isoformat()
        title = self._title_from_messages(clean_messages)
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO conversations(session_id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET title = excluded.title, "
                    "updated_at = excluded.updated_at",
                    (session_id, title, now, now),
                )
                connection.execute(
                    "DELETE FROM messages WHERE session_id = ?", (session_id,)
                )
                connection.executemany(
                    "INSERT INTO messages(session_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (session_id, message["role"], message["content"], now)
                        for message in clean_messages
                    ],
                )

    def append_exchange(self, session_id, user_message, assistant_message):
        now = datetime.now(timezone.utc).isoformat()
        title = self._format_title(user_message)
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO conversations(session_id, title, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at",
                    (session_id, title, now, now),
                )
                connection.executemany(
                    "INSERT INTO messages(session_id, role, content, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        (session_id, "user", user_message, now),
                        (session_id, "assistant", assistant_message, now),
                    ],
                )

    def append_request_metrics(self, session_id, model_key, report, cost_usd):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO request_metrics("
                    "session_id, model_key, current_request_tokens, history_tokens, "
                    "instructions_tokens, estimated_input_tokens, input_tokens, "
                    "output_tokens, total_tokens, cost_usd, context_limit, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        model_key,
                        report["current_request_tokens"],
                        report["history_tokens"],
                        report["instructions_tokens"],
                        report["estimated_input_tokens"],
                        report["api_input_tokens"],
                        report["response_tokens"],
                        report["api_total_tokens"],
                        cost_usd,
                        report["context_limit"],
                        now,
                    ),
                )

    def request_metrics(self, session_id):
        with self._lock:
            with self._connection() as connection:
                rows = connection.execute(
                    "SELECT current_request_tokens, history_tokens, instructions_tokens, "
                    "estimated_input_tokens, input_tokens, output_tokens, total_tokens, "
                    "cost_usd, context_limit, created_at FROM request_metrics "
                    "WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
        return [
            {
                "turn": index,
                "current_request_tokens": row[0],
                "history_tokens": row[1],
                "instructions_tokens": row[2],
                "estimated_input_tokens": row[3],
                "api_input_tokens": row[4],
                "response_tokens": row[5],
                "api_total_tokens": row[6],
                "cost_usd": row[7],
                "context_limit": row[8],
                "created_at": row[9],
            }
            for index, row in enumerate(rows, 1)
        ]

    def usage_summary(self, session_id):
        turns = self.request_metrics(session_id)
        return {
            "turn_count": len(turns),
            "cumulative_api_tokens": sum(turn["api_total_tokens"] for turn in turns),
            "cumulative_input_tokens": sum(turn["api_input_tokens"] for turn in turns),
            "cumulative_output_tokens": sum(turn["response_tokens"] for turn in turns),
            "cumulative_cost_usd": round(sum(turn["cost_usd"] for turn in turns), 8),
            "turns": turns,
        }

    def clear(self, session_id):
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "DELETE FROM conversations WHERE session_id = ?", (session_id,)
                )

    def replace_last_assistant(self, session_id, answer):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "UPDATE messages SET content = ?, created_at = ? "
                    "WHERE id = (SELECT id FROM messages WHERE session_id = ? "
                    "AND role = 'assistant' ORDER BY id DESC LIMIT 1)",
                    (answer, now, session_id),
                )
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE session_id = ?",
                    (now, session_id),
                )

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self):
        with self._lock:
            with self._connection() as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS conversations ("
                    "session_id TEXT PRIMARY KEY, title TEXT NOT NULL, "
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS messages ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "session_id TEXT NOT NULL REFERENCES conversations(session_id) "
                    "ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user', 'assistant')), "
                    "content TEXT NOT NULL, created_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_messages_session_id_id "
                    "ON messages(session_id, id)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS request_metrics ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "session_id TEXT NOT NULL REFERENCES conversations(session_id) "
                    "ON DELETE CASCADE, model_key TEXT NOT NULL, "
                    "current_request_tokens INTEGER NOT NULL, "
                    "history_tokens INTEGER NOT NULL, instructions_tokens INTEGER NOT NULL, "
                    "estimated_input_tokens INTEGER NOT NULL, input_tokens INTEGER NOT NULL, "
                    "output_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL, "
                    "cost_usd REAL NOT NULL, context_limit INTEGER NOT NULL, "
                    "created_at TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_request_metrics_session_id_id "
                    "ON request_metrics(session_id, id)"
                )
                connection.execute("PRAGMA optimize")

    def _migrate_legacy_json(self):
        if not self.legacy_json_path or not self.legacy_json_path.exists():
            return
        try:
            conversations = json.loads(
                self.legacy_json_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(conversations, dict):
            return
        for session_id, entry in conversations.items():
            messages = entry.get("messages", []) if isinstance(entry, dict) else entry
            if not isinstance(messages, list) or not messages:
                continue
            with self._connection() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM conversations WHERE session_id = ?", (session_id,)
                ).fetchone()
            if not exists:
                self.save(session_id, messages)

    @classmethod
    def _title_from_messages(cls, messages):
        first_user_message = next(
            (
                message["content"]
                for message in messages
                if message["role"] == "user"
            ),
            "Новый диалог",
        )
        return cls._format_title(first_user_message)

    @staticmethod
    def _format_title(message):
        compact = " ".join(message.split())
        return compact[:48] + ("…" if len(compact) > 48 else "")


class SimpleAgent:
    """Encapsulates one LLM model, its HTTP request, and response parsing."""

    def __init__(
        self,
        model_key="deepseek-v4-pro",
        api_key=None,
        session_id=None,
        history_database=HISTORY_DATABASE,
    ):
        if model_key not in MODEL_OPTIONS:
            raise ValueError("Неизвестная модель агента.")
        self.model_key = model_key
        self.config = MODEL_OPTIONS[model_key]
        self.api_key = api_key or os.environ.get(self.config["api_key_env"])
        self.session_id = session_id
        self.store = ConversationStore(history_database) if session_id else None
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
        token_report = self.token_report(user_request, messages, instructions)
        messages.append({"role": "user", "content": user_request})
        result = self.run(messages, instructions, temperature, token_report=token_report)
        self.store.append_exchange(self.session_id, user_request, result["text"])
        result["token_report"]["conversation_tokens_after"] = estimate_messages_tokens(
            self.history
        )
        self.store.append_request_metrics(
            self.session_id, self.model_key, result["token_report"], result["cost_usd"]
        )
        result["conversation_totals"] = self.store.usage_summary(self.session_id)
        return result

    def clear_history(self):
        if self.store:
            self.store.clear(self.session_id)

    def replace_last_answer(self, answer):
        """Keep stored history identical to the answer shown in the interface."""
        if not self.store:
            return
        self.store.replace_last_assistant(self.session_id, answer)

    def run(self, user_request, instructions=None, temperature=None, token_report=None):
        """Send a request to the LLM and return text plus execution metrics."""
        if token_report is None:
            current_text = (
                "\n".join(message.get("content", "") for message in user_request)
                if isinstance(user_request, list)
                else user_request
            )
            token_report = self.token_report(current_text, [], instructions)
        if token_report["estimated_input_tokens"] > token_report["context_limit"]:
            token_report["status"] = "overflow"
            raise ContextWindowExceeded(token_report)
        request_body = self._build_request_body(
            user_request, instructions, temperature
        )
        result, elapsed_seconds = self._send(request_body)
        usage = result.get("usage") or {}
        cost_usd, pricing_note = self._estimate_cost(usage)
        text = self._extract_text(result)
        token_report.update(
            {
                "api_input_tokens": int(usage.get("input_tokens") or 0),
                "response_tokens": int(usage.get("output_tokens") or 0),
                "api_total_tokens": int(usage.get("total_tokens") or 0),
                "response_tokens_estimated": estimate_text_tokens(text),
            }
        )
        return {
            "text": text,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "usage": {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            "cost_usd": round(cost_usd, 8),
            "pricing_note": pricing_note,
            "token_report": token_report,
        }

    def token_report(self, user_request, history=None, instructions=None):
        history = history or []
        current_tokens = estimate_text_tokens(user_request) + 4
        history_tokens = estimate_messages_tokens(history)
        instructions_tokens = estimate_text_tokens(instructions) + (4 if instructions else 0)
        estimated_input_tokens = current_tokens + history_tokens + instructions_tokens + 2
        context_limit = int(
            os.environ.get(
                self.config["context_env"],
                self.config["context_window"],
            )
        )
        usage_percent = round(estimated_input_tokens / context_limit * 100, 2)
        return {
            "current_request_tokens": current_tokens,
            "history_tokens": history_tokens,
            "instructions_tokens": instructions_tokens,
            "estimated_input_tokens": estimated_input_tokens,
            "api_input_tokens": 0,
            "response_tokens": 0,
            "api_total_tokens": 0,
            "context_limit": context_limit,
            "context_remaining_tokens": max(0, context_limit - estimated_input_tokens),
            "context_usage_percent": usage_percent,
            "status": "overflow" if estimated_input_tokens > context_limit else (
                "warning" if usage_percent >= 80 else "ok"
            ),
            "counting_note": (
                "Запрос и история рассчитаны локально приблизительно; "
                "общий вход и ответ после запроса берутся из точных данных API."
            ),
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
