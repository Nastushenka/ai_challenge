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


def call_model(api_key, input_value, instructions=None, temperature=None):
    request_body = {
        "model": os.environ.get("LLM_MODEL", "deepseek-v4-pro"),
        "input": input_value,
    }
    if instructions:
        request_body["instructions"] = instructions
    if temperature is not None:
        request_body["temperature"] = temperature
        request_body["reasoning"] = {"effort": "none"}
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
    for attempt in range(2):
        try:
            with urlopen(request, timeout=90, context=ssl_context) as response:
                result = json.load(response)
            break
        except HTTPError as error:
            if attempt == 0 and error.code in {429, 500, 502, 503, 504}:
                continue
            raise
    return "".join(
        content.get("text", "")
        for item in result.get("output", [])
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )


def compare_solutions(api_key, prompt, max_words, temperature=None):
    def direct_solution():
        return call_model(api_key, prompt, temperature=temperature)

    def step_by_step_solution():
        return call_model(
            api_key,
            prompt,
            "Решай пошагово и показывай проверяемую логику решения. "
            "Структура ответа: 1) кратко сформулируй цель; 2) перечисли исходные "
            "факты и ограничения; 3) выполни пронумерованные шаги, объясняя, почему "
            "каждый шаг следует из условий; 4) проверь результат по всем ограничениям "
            "и рассмотри возможную альтернативу; 5) отдельно сформулируй окончательный "
            "ответ. Не выдумывай отсутствующие данные: явно отмечай неоднозначность. "
            f"Отвечай на русском языке и не превышай {max_words} слов.",
            temperature,
        )

    def prompt_engineering_solution():
        generated_prompt = call_model(
            api_key,
            prompt,
            "Ты — промпт-инженер. Преобразуй задачу пользователя в точный, "
            "самодостаточный промпт для другой языковой модели. Не решай исходную "
            "задачу и не подсказывай ответ. Верни только готовый промпт со следующими "
            "разделами: «Роль», «Задача», «Исходные данные», «Ограничения», "
            "«Метод решения и самопроверки», «Формат ответа». Сохрани все значимые "
            "детали исходной задачи. Потребуй проверить каждое условие, отделить факты "
            "от предположений, рассмотреть альтернативы и дать однозначный итог. "
            "Промпт должен быть на русском языке и подходить для использования без "
            "дополнительного контекста.",
            temperature,
        )
        answer = call_model(api_key, generated_prompt, temperature=temperature)
        return generated_prompt, answer

    def expert_group_solution():
        return call_model(
            api_key,
            prompt,
            "Создай группу из трёх экспертов: аналитика, инженера и критика. "
            "Пусть каждый независимо предложит своё решение задачи и объяснит ход мысли. "
            f"Чётко раздели ответы экспертов. Отвечай на русском языке, до {max_words} слов.",
            temperature,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        direct_future = executor.submit(direct_solution)
        steps_future = executor.submit(step_by_step_solution)
        prompt_future = executor.submit(prompt_engineering_solution)
        experts_future = executor.submit(expert_group_solution)
        generated_prompt, prompt_answer = prompt_future.result()
        solutions = [
            {
                "id": "direct",
                "title": "1. Прямой ответ",
                "description": "Только исходная задача, без дополнительных инструкций.",
                "answer": direct_future.result(),
            },
            {
                "id": "steps",
                "title": "2. Проверяемое пошаговое решение",
                "description": "Факты, ограничения, обоснованные шаги, проверка и итог.",
                "answer": steps_future.result(),
            },
            {
                "id": "prompt",
                "title": "3. Решение через улучшенный промпт",
                "description": "Сначала создаётся самодостаточный промпт, затем он решает задачу.",
                "generated_prompt": generated_prompt,
                "answer": prompt_answer,
            },
            {
                "id": "experts",
                "title": "4. Группа экспертов",
                "description": "Независимые позиции аналитика, инженера и критика.",
                "answer": experts_future.result(),
            },
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
        temperature,
    )
    return {"solutions": solutions, "analysis": truncate_words(analysis, max_words)}


def compare_temperatures(api_key, prompt, max_words):
    settings = [
        ("temperature-0", "Temperature = 0", 0.0, "Максимальная стабильность и фокус."),
        ("temperature-07", "Temperature = 0.7", 0.7, "Баланс точности и вариативности."),
        ("temperature-12", "Temperature = 1.2", 1.2, "Больше разнообразия и неожиданных идей."),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            value: executor.submit(call_model, api_key, prompt, None, value)
            for _, _, value, _ in settings
        }
        solutions = [
            {
                "id": solution_id,
                "title": title,
                "description": description,
                "temperature": value,
                "answer": truncate_words(futures[value].result(), max_words),
            }
            for solution_id, title, value, description in settings
        ]

    comparison_text = "\n\n".join(
        f"{solution['title']}\n{solution['answer']}" for solution in solutions
    )
    analysis = call_model(
        api_key,
        f"Исходная задача:\n{prompt}\n\nОтветы:\n{comparison_text}",
        "Сравни три ответа, созданные с разными значениями temperature. "
        "Проанализируй каждый по критериям: 1) точность, 2) креативность, "
        "3) разнообразие идей и формулировок. Отдельно отметь сильные и слабые "
        "стороны каждого значения. В конце сделай практический вывод, для каких "
        "типов задач лучше подходят temperature 0, 0.7 и 1.2. Отвечай на русском "
        f"языке, структурированно и не превышай {max_words} слов.",
        0.0,
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
            compare_temperature_mode = payload.get("compare_temperature_mode", False) is True
            use_temperature = payload.get("use_temperature", False) is True
            temperature = float(payload.get("temperature", 1.0)) if use_temperature else None
            use_prompt_limit = payload.get("use_prompt_limit", False) is True
            max_prompt_chars = (
                int(payload.get("max_prompt_chars", 1000)) if use_prompt_limit else 12000
            )
            finish_mode = payload.get("finish_mode", "none")
            finish_value = payload.get("finish_value", "").strip()
            history = payload.get("history", [])
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Некорректный запрос."})
            return

        if not prompt:
            self.send_json(400, {"error": "Введите запрос."})
            return
        if use_prompt_limit and not 1 <= max_prompt_chars <= 12000:
            self.send_json(
                400,
                {"error": "Ограничение запроса должно быть от 1 до 12 000 символов."},
            )
            return
        if len(prompt) > max_prompt_chars:
            self.send_json(
                400,
                {"error": f"Запрос не должен превышать {max_prompt_chars} символов."},
            )
            return
        if not 20 <= max_words <= 2000:
            self.send_json(400, {"error": "Укажите ограничение от 20 до 2000 слов."})
            return
        if temperature is not None and not 0 <= temperature <= 2:
            self.send_json(400, {"error": "Temperature должна быть от 0 до 2."})
            return
        if compare_mode and compare_temperature_mode:
            self.send_json(400, {"error": "Выберите только один режим сравнения."})
            return
        if compare_temperature_mode:
            try:
                comparison = compare_temperatures(api_key, prompt, max_words)
            except HTTPError as error:
                try:
                    message = json.load(error).get("error", {}).get("message")
                except Exception:
                    message = None
                self.send_json(error.code, {"error": message or "Ошибка API модели."})
                return
            except Exception:
                self.send_json(502, {"error": "Не удалось сравнить значения temperature."})
                return
            self.send_json(200, comparison)
            return
        if compare_mode:
            try:
                comparison = compare_solutions(api_key, prompt, max_words, temperature)
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
                temperature,
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
