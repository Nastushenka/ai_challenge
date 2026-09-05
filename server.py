import json
import os
import re
import ssl
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")


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


def truncate_words(text, max_words):
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return text
    cut_at = words[max_words - 1].end()
    return text[:cut_at].rstrip(" ,;:.") + "…"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            self.send_json(503, {"error": "На сервере не настроен LLM_API_KEY."})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            prompt = payload.get("prompt", "").strip()
            use_format = payload.get("use_format", True) is not False
            max_words = int(payload.get("max_words", 200))
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Некорректный запрос."})
            return

        if not prompt or len(prompt) > 12000:
            self.send_json(400, {"error": "Введите от 1 до 12 000 символов."})
            return
        if not 20 <= max_words <= 2000:
            self.send_json(400, {"error": "Укажите ограничение от 20 до 2000 слов."})
            return

        request_body = {
            "model": os.environ.get("LLM_MODEL", "deepseek-v4-pro"),
            "input": prompt,
        }
        instructions = [
            f"Отвечай на русском языке. Весь ответ, включая заголовки, должен "
            f"содержать не более {max_words} слов."
        ]
        if use_format:
            instructions.append(
                "Строго соблюдай формат из трёх блоков. "
                "1) Заголовок «Краткий ответ:» и один-два предложения. "
                "2) Заголовок «Основные пункты:» и нумерованный список из двух-пяти пунктов. "
                "3) Заголовок «Итог:» и одно заключительное предложение. "
                "Не используй лишние вступления."
            )
        request_body["instructions"] = " ".join(instructions)
        body = json.dumps(request_body).encode()
        api_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/")
        request = Request(
            f"{api_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            ssl_context = ssl.create_default_context(
                cafile=str(SYSTEM_CA_FILE) if SYSTEM_CA_FILE.exists() else None
            )
            with urlopen(request, timeout=90, context=ssl_context) as response:
                result = json.load(response)
        except HTTPError as error:
            try:
                message = json.load(error).get("error", {}).get("message")
            except Exception:
                message = None
            self.send_json(error.code, {"error": message or "Ошибка API модели."})
            return
        except Exception:
            self.send_json(502, {"error": "Не удалось связаться с API модели."})
            return

        answer = "".join(
            content.get("text", "")
            for item in result.get("output", [])
            if item.get("type") == "message"
            for content in item.get("content", [])
            if content.get("type") == "output_text"
        )
        answer = truncate_words(answer, max_words)
        self.send_json(200, {"answer": answer})

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print("Nastia Chat: http://127.0.0.1:8000")
    ThreadingHTTPServer(address, Handler).serve_forever()
