import json
import os
import re
import ssl
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
FINAL_MARKER = "[[READY]]"


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


def call_model(api_key, input_value, instructions=None):
    request_body = {
        "model": os.environ.get("LLM_MODEL", "deepseek-v4-pro"),
        "input": input_value,
    }
    if instructions:
        request_body["instructions"] = instructions
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
    ssl_context = ssl.create_default_context(
        cafile=str(SYSTEM_CA_FILE) if SYSTEM_CA_FILE.exists() else None
    )
    with urlopen(request, timeout=90, context=ssl_context) as response:
        result = json.load(response)
    return "".join(
        content.get("text", "")
        for item in result.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )


def compare_solutions(api_key, prompt, max_words):
    def direct_solution():
        return call_model(api_key, prompt)

    def step_by_step_solution():
        return call_model(
            api_key,
            prompt,
            f"Решай пошагово. Отвечай на русском языке и не превышай {max_words} слов.",
        )

    def prompt_engineering_solution():
        generated_prompt = call_model(
            api_key,
            prompt,
            "Составь на русском языке точный и самодостаточный промпт для решения "
            "задачи пользователя. Верни только готовый промпт без пояснений и решения.",
        )
        answer = call_model(api_key, generated_prompt)
        return generated_prompt, answer

    def expert_group_solution():
        return call_model(
            api_key,
            prompt,
            "Создай группу из трёх экспертов: аналитика, инженера и критика. "
            "Пусть каждый независимо предложит своё решение задачи и объяснит ход мысли. "
            f"Чётко раздели ответы экспертов. Отвечай на русском языке, до {max_words} слов.",
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        direct_future = executor.submit(direct_solution)
        steps_future = executor.submit(step_by_step_solution)
        prompt_future = executor.submit(prompt_engineering_solution)
        experts_future = executor.submit(expert_group_solution)
        generated_prompt, prompt_answer = prompt_future.result()
        solutions = [
            {"id": "direct", "title": "1. Прямой ответ", "answer": direct_future.result()},
            {"id": "steps", "title": "2. Решение пошагово", "answer": steps_future.result()},
            {
                "id": "prompt",
                "title": "3. Сначала создать промпт",
                "generated_prompt": generated_prompt,
                "answer": prompt_answer,
            },
            {"id": "experts", "title": "4. Группа экспертов", "answer": experts_future.result()},
        ]

    for solution in solutions:
        solution["answer"] = truncate_words(solution["answer"], max_words)

    comparison_text = "\n\n".join(
        f"{solution['title']}\n{solution['answer']}" for solution in solutions
    )
    analysis = call_model(
        api_key,
        f"Исходная задача:\n{prompt}\n\nПолученные решения:\n{comparison_text}",
        "Проанализируй четыре решения на русском языке. Сравни их корректность, "
        "полноту, понятность и надёжность. Укажи совпадения и противоречия, выбери "
        f"лучший подход и сформулируй итоговый вывод. Не превышай {max_words} слов.",
    )
    return {"solutions": solutions, "analysis": truncate_words(analysis, max_words)}


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
            compare_mode = payload.get("compare_mode", False) is True
            finish_mode = payload.get("finish_mode", "none")
            finish_value = payload.get("finish_value", "").strip()
            history = payload.get("history", [])
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Некорректный запрос."})
            return

        if not prompt or len(prompt) > 12000:
            self.send_json(400, {"error": "Введите от 1 до 12 000 символов."})
            return
        if not 20 <= max_words <= 2000:
            self.send_json(400, {"error": "Укажите ограничение от 20 до 2000 слов."})
            return
        if compare_mode:
            try:
                comparison = compare_solutions(api_key, prompt, max_words)
            except HTTPError as error:
                try:
                    message = json.load(error).get("error", {}).get("message")
                except Exception:
                    message = None
                self.send_json(error.code, {"error": message or "Ошибка API модели."})
                return
            except Exception:
                self.send_json(502, {"error": "Не удалось получить все варианты решения."})
                return
            self.send_json(200, comparison)
            return
        if finish_mode not in {"none", "instruction", "sequence", "dialogue"}:
            self.send_json(400, {"error": "Выберите допустимый режим завершения."})
            return
        if finish_mode != "none" and not finish_value:
            self.send_json(400, {"error": "Укажите условие завершения ответа."})
            return
        max_finish_length = 100 if finish_mode == "sequence" else 500
        if len(finish_value) > max_finish_length:
            self.send_json(
                400,
                {"error": f"Условие завершения не должно превышать {max_finish_length} символов."},
            )
            return
        if not isinstance(history, list) or len(history) > 20:
            self.send_json(400, {"error": "История диалога слишком длинная."})
            return

        messages = []
        for item in history:
            if not isinstance(item, dict):
                self.send_json(400, {"error": "Некорректная история диалога."})
                return
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                self.send_json(400, {"error": "Некорректная история диалога."})
                return
            content = content.strip()
            if not content or len(content) > 12000:
                self.send_json(400, {"error": "Некорректная история диалога."})
                return
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": prompt})

        dialogue_mode = finish_mode == "dialogue"
        instructions = [
            f"Отвечай на русском языке. Весь ответ, включая заголовки, должен "
            f"содержать не более {max_words} слов."
        ]
        if dialogue_mode:
            instructions.append(
                "Веди диалог до готовности результата. Если данных недостаточно, "
                "задай ровно один короткий уточняющий вопрос и не давай итоговый ответ. "
                f"Критерий готовности: {finish_value} "
                f"Когда критерий выполнен, начни ответ с маркера {FINAL_MARKER}, "
                "затем сразу дай окончательный результат. Не используй маркер раньше "
                "и не задавай после итогового результата вопросов."
            )
        elif finish_mode == "instruction":
            instructions.append(f"Условие завершения ответа: {finish_value}")
        elif finish_mode == "sequence":
            instructions.append(
                "Заверши ответ точной последовательностью "
                f"{json.dumps(finish_value, ensure_ascii=False)}. "
                "После неё ничего не добавляй."
            )
        if use_format:
            instructions.append(
                "Если ты задаёшь уточняющий вопрос, не применяй к нему шаблон ответа. "
                "Для окончательного результата строго соблюдай формат из трёх блоков. "
                "1) Заголовок «Краткий ответ:» и один-два предложения. "
                "2) Заголовок «Основные пункты:» и нумерованный список из двух-пяти пунктов. "
                "3) Заголовок «Итог:» и одно заключительное предложение. "
                "Не используй лишние вступления."
            )
        try:
            answer = call_model(
                api_key,
                messages if dialogue_mode else prompt,
                " ".join(instructions),
            )
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

        complete = dialogue_mode and FINAL_MARKER in answer
        if complete:
            answer = answer.replace(FINAL_MARKER, "", 1).strip()
        elif finish_mode == "sequence" and finish_value in answer:
            answer = answer.split(finish_value, 1)[0].rstrip()
            complete = True
        answer = truncate_words(answer, max_words)
        self.send_json(200, {"answer": answer, "complete": complete})

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
