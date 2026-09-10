import json
import re
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from agent import (
    AgentConfigurationError,
    ContextWindowExceeded,
    ConversationStore,
    MODEL_OPTIONS,
    SimpleAgent,
    estimate_messages_tokens,
)
from token_scenarios import analyze_prepared_dialogues


ROOT = Path(__file__).parent
FINAL_MARKER = "[[READY]]"
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def truncate_words(text, max_words):
    if max_words is None:
        return text
    words = list(re.finditer(r"\S+", text))
    if len(words) <= max_words:
        return text
    cut_at = words[max_words - 1].end()
    return text[:cut_at].rstrip(" ,;:.") + "…"


def result_metrics(result):
    usage = result["usage"]
    metrics = {
        "elapsed_seconds": result["elapsed_seconds"],
        "input_tokens": usage["input_tokens"],
        "output_tokens": usage["output_tokens"],
        "total_tokens": usage["total_tokens"],
        "cost_usd": result["cost_usd"],
        "pricing_note": result["pricing_note"],
    }
    if result.get("token_report"):
        metrics["token_report"] = result["token_report"]
    if result.get("conversation_totals"):
        metrics["conversation_totals"] = result["conversation_totals"]
    return metrics


def compare_solutions(
    prompt, max_words, temperature=None, model_key="deepseek-v4-pro"
):
    agent = SimpleAgent(model_key)
    limit_instruction = (
        f" Не превышай {max_words} слов." if max_words is not None else ""
    )

    def direct_solution():
        return agent.run(prompt, temperature=temperature)

    def step_by_step_solution():
        return agent.run(
            prompt,
            "Решай пошагово и показывай проверяемую логику решения. "
            "Структура ответа: 1) кратко сформулируй цель; 2) перечисли исходные "
            "факты и ограничения; 3) выполни пронумерованные шаги, объясняя, почему "
            "каждый шаг следует из условий; 4) проверь результат по всем ограничениям "
            "и рассмотри возможную альтернативу; 5) отдельно сформулируй окончательный "
            "ответ. Не выдумывай отсутствующие данные: явно отмечай неоднозначность. "
            f"Отвечай на русском языке.{limit_instruction}",
            temperature,
        )

    def prompt_engineering_solution():
        generated_prompt_result = agent.run(
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
        generated_prompt = generated_prompt_result["text"]
        answer_result = agent.run(generated_prompt, temperature=temperature)
        combined = {
            "elapsed_seconds": round(
                generated_prompt_result["elapsed_seconds"]
                + answer_result["elapsed_seconds"],
                3,
            ),
            "usage": {
                key: generated_prompt_result["usage"][key]
                + answer_result["usage"][key]
                for key in ("input_tokens", "output_tokens", "total_tokens")
            },
            "cost_usd": round(
                generated_prompt_result["cost_usd"] + answer_result["cost_usd"],
                8,
            ),
            "pricing_note": answer_result["pricing_note"],
        }
        return generated_prompt, answer_result, combined

    def expert_group_solution():
        return agent.run(
            prompt,
            "Создай группу из трёх экспертов: аналитика, инженера и критика. "
            "Пусть каждый независимо предложит своё решение задачи и объяснит ход мысли. "
            f"Чётко раздели ответы экспертов. Отвечай на русском языке.{limit_instruction}",
            temperature,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        direct_future = executor.submit(direct_solution)
        steps_future = executor.submit(step_by_step_solution)
        prompt_future = executor.submit(prompt_engineering_solution)
        experts_future = executor.submit(expert_group_solution)
        generated_prompt, prompt_answer, prompt_metrics = prompt_future.result()
        direct_result = direct_future.result()
        steps_result = steps_future.result()
        experts_result = experts_future.result()
        solutions = [
            {
                "id": "direct",
                "title": "1. Прямой ответ",
                "description": "Только исходная задача, без дополнительных инструкций.",
                "answer": direct_result["text"],
                "metrics": result_metrics(direct_result),
            },
            {
                "id": "steps",
                "title": "2. Проверяемое пошаговое решение",
                "description": "Факты, ограничения, обоснованные шаги, проверка и итог.",
                "answer": steps_result["text"],
                "metrics": result_metrics(steps_result),
            },
            {
                "id": "prompt",
                "title": "3. Решение через улучшенный промпт",
                "description": "Сначала создаётся самодостаточный промпт, затем он решает задачу.",
                "generated_prompt": generated_prompt,
                "answer": prompt_answer["text"],
                "metrics": result_metrics(prompt_metrics),
            },
            {
                "id": "experts",
                "title": "4. Группа экспертов",
                "description": "Независимые позиции аналитика, инженера и критика.",
                "answer": experts_result["text"],
                "metrics": result_metrics(experts_result),
            },
        ]

    for solution in solutions:
        solution["answer"] = truncate_words(solution["answer"], max_words)

    comparison_text = "\n\n".join(
        f"{solution['title']}\n{solution['answer']}" for solution in solutions
    )
    analysis_result = agent.run(
        f"Исходная задача:\n{prompt}\n\nПолученные решения:\n{comparison_text}",
        "Проанализируй четыре решения на русском языке. Сравни их корректность, "
        "полноту, понятность и надёжность. Укажи совпадения и противоречия, выбери "
        f"лучший подход и сформулируй итоговый вывод.{limit_instruction}",
        temperature,
    )
    return {
        "solutions": solutions,
        "analysis": truncate_words(analysis_result["text"], max_words),
        "analysis_metrics": result_metrics(analysis_result),
    }


def compare_temperatures(prompt, max_words, model_key="deepseek-v4-pro"):
    agent = SimpleAgent(model_key)
    limit_instruction = (
        f" Не превышай {max_words} слов." if max_words is not None else ""
    )
    settings = [
        ("temperature-0", "Temperature = 0", 0.0, "Максимальная стабильность и фокус."),
        ("temperature-07", "Temperature = 0.7", 0.7, "Баланс точности и вариативности."),
        ("temperature-12", "Temperature = 1.2", 1.2, "Больше разнообразия и неожиданных идей."),
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            value: executor.submit(agent.run, prompt, None, value)
            for _, _, value, _ in settings
        }
        solutions = [
            {
                "id": solution_id,
                "title": title,
                "description": description,
                "temperature": value,
                "answer": truncate_words(futures[value].result()["text"], max_words),
                "metrics": result_metrics(futures[value].result()),
            }
            for solution_id, title, value, description in settings
        ]

    comparison_text = "\n\n".join(
        f"{solution['title']}\n{solution['answer']}" for solution in solutions
    )
    analysis_result = agent.run(
        f"Исходная задача:\n{prompt}\n\nОтветы:\n{comparison_text}",
        "Сравни три ответа, созданные с разными значениями temperature. "
        "Проанализируй каждый по критериям: 1) точность, 2) креативность, "
        "3) разнообразие идей и формулировок. Отдельно отметь сильные и слабые "
        "стороны каждого значения. В конце сделай практический вывод, для каких "
        "типов задач лучше подходят temperature 0, 0.7 и 1.2. Отвечай на русском "
        f"языке и структурированно.{limit_instruction}",
        0.0,
    )
    return {
        "solutions": solutions,
        "analysis": truncate_words(analysis_result["text"], max_words),
        "analysis_metrics": result_metrics(analysis_result),
    }


def compare_models(prompt, max_words, temperature=None):
    limit_instruction = (
        f" Весь ответ должен содержать не более {max_words} слов."
        if max_words is not None
        else ""
    )
    instructions = f"Отвечай на русском языке.{limit_instruction}"

    def run_model(model_key):
        result = SimpleAgent(model_key).run(prompt, instructions, temperature)
        result["text"] = truncate_words(result["text"], max_words)
        return model_key, result

    with ThreadPoolExecutor(max_workers=len(MODEL_OPTIONS)) as executor:
        futures = [executor.submit(run_model, model_key) for model_key in MODEL_OPTIONS]
        results = dict(future.result() for future in futures)

    solutions = []
    for model_key, model_config in MODEL_OPTIONS.items():
        result = results[model_key]
        solutions.append(
            {
                "id": model_key,
                "title": model_config["label"],
                "description": "Один и тот же запрос без специальных подсказок для модели.",
                "answer": result["text"],
                "metrics": result_metrics(result),
            }
        )

    comparison_text = "\n\n".join(
        f"{solution['title']}\n"
        f"Время: {solution['metrics']['elapsed_seconds']} сек.\n"
        f"Токены: {solution['metrics']['total_tokens']}\n"
        f"Стоимость: ${solution['metrics']['cost_usd']:.8f}\n"
        f"Ответ:\n{solution['answer']}"
        for solution in solutions
    )
    analysis_result = SimpleAgent("deepseek-v4-pro").run(
        f"Исходный запрос:\n{prompt}\n\nРезультаты моделей:\n{comparison_text}",
        "Сравни ответы трёх моделей на русском языке. Оцени: 1) качество и "
        "корректность ответа, 2) скорость по измеренному времени, 3) ресурсоёмкость "
        "по количеству токенов и расчётной стоимости. Назови победителя по каждому "
        "критерию, объясни компромиссы и закончи отдельным практическим выводом о "
        "том, какую модель выбрать для подобных запросов. Не выдумывай метрики и "
        "используй только приведённые значения."
        + limit_instruction,
        0.0,
    )
    return {
        "solutions": solutions,
        "analysis": truncate_words(analysis_result["text"], max_words),
        "analysis_metrics": result_metrics(analysis_result),
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/conversations":
            self.send_json(200, {"conversations": ConversationStore().list()})
            return
        if parsed.path == "/api/token-tests":
            model_key = parse_qs(parsed.query).get("model", ["deepseek-v4-pro"])[0]
            if model_key not in MODEL_OPTIONS:
                self.send_json(400, {"error": "Выберите доступную модель."})
                return
            self.send_json(
                200,
                {
                    "model": model_key,
                    "model_label": MODEL_OPTIONS[model_key]["label"],
                    "scenarios": analyze_prepared_dialogues(model_key),
                },
            )
            return
        if parsed.path != "/api/history":
            super().do_GET()
            return
        session_id = parse_qs(parsed.query).get("session_id", [""])[0]
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            self.send_json(400, {"error": "Некорректный идентификатор диалога."})
            return
        store = ConversationStore()
        history = store.get(session_id)
        summary = store.usage_summary(session_id)
        summary["dialogue_tokens"] = estimate_messages_tokens(history)
        self.send_json(200, {"history": history, "token_summary": summary})

    def do_POST(self):
        if self.path == "/api/reset":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                session_id = payload.get("session_id", "")
            except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
                self.send_json(400, {"error": "Некорректный запрос."})
                return
            if not SESSION_ID_PATTERN.fullmatch(session_id):
                self.send_json(400, {"error": "Некорректный идентификатор диалога."})
                return
            ConversationStore().clear(session_id)
            self.send_json(200, {"cleared": True})
            return

        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            prompt = payload.get("prompt", "").strip()
            model_key = payload.get("model", "deepseek-v4-pro")
            use_format = payload.get("use_format", True) is not False
            use_word_limit = payload.get("use_word_limit", False) is True
            max_words = int(payload.get("max_words", 200)) if use_word_limit else None
            compare_mode = payload.get("compare_mode", False) is True
            compare_temperature_mode = payload.get("compare_temperature_mode", False) is True
            compare_models_mode = payload.get("compare_models_mode", False) is True
            use_temperature = payload.get("use_temperature", False) is True
            temperature = float(payload.get("temperature", 1.0)) if use_temperature else None
            use_prompt_limit = payload.get("use_prompt_limit", False) is True
            max_prompt_chars = (
                int(payload.get("max_prompt_chars", 1000)) if use_prompt_limit else 12000
            )
            finish_mode = payload.get("finish_mode", "none")
            finish_value = payload.get("finish_value", "").strip()
            session_id = payload.get("session_id", "")
        except (TypeError, ValueError, AttributeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Некорректный запрос."})
            return

        if not prompt:
            self.send_json(400, {"error": "Введите запрос."})
            return
        if model_key not in MODEL_OPTIONS:
            self.send_json(400, {"error": "Выберите доступную модель."})
            return
        if not SESSION_ID_PATTERN.fullmatch(session_id):
            self.send_json(400, {"error": "Некорректный идентификатор диалога."})
            return
        try:
            selected_agent = SimpleAgent(model_key, session_id=session_id)
        except AgentConfigurationError as error:
            self.send_json(503, {"error": str(error)})
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
        if use_word_limit and not 20 <= max_words <= 2000:
            self.send_json(400, {"error": "Укажите ограничение от 20 до 2000 слов."})
            return
        if temperature is not None and not 0 <= temperature <= 2:
            self.send_json(400, {"error": "Temperature должна быть от 0 до 2."})
            return
        if sum((compare_mode, compare_temperature_mode, compare_models_mode)) > 1:
            self.send_json(400, {"error": "Выберите только один режим сравнения."})
            return
        if compare_models_mode:
            try:
                for candidate in MODEL_OPTIONS:
                    SimpleAgent(candidate)
            except AgentConfigurationError as error:
                self.send_json(503, {"error": str(error)})
                return
            try:
                comparison = compare_models(prompt, max_words, temperature)
            except ContextWindowExceeded as error:
                self.send_json(413, {"error": str(error), "token_report": error.report})
                return
            except HTTPError as error:
                try:
                    message = json.load(error).get("error", {}).get("message")
                except Exception:
                    message = None
                self.send_json(error.code, {"error": message or "Ошибка API модели."})
                return
            except Exception:
                self.send_json(502, {"error": "Не удалось сравнить все модели."})
                return
            self.send_json(200, comparison)
            return
        if compare_temperature_mode:
            try:
                comparison = compare_temperatures(prompt, max_words, model_key)
            except ContextWindowExceeded as error:
                self.send_json(413, {"error": str(error), "token_report": error.report})
                return
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
                comparison = compare_solutions(
                    prompt, max_words, temperature, model_key
                )
            except ContextWindowExceeded as error:
                self.send_json(413, {"error": str(error), "token_report": error.report})
                return
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
        dialogue_mode = finish_mode == "dialogue"
        instructions = ["Отвечай на русском языке."]
        if max_words is not None:
            instructions.append(
                f"Весь ответ, включая заголовки, должен содержать не более "
                f"{max_words} слов."
            )
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
            agent_result = selected_agent.chat(
                prompt,
                " ".join(instructions),
                temperature,
            )
        except ContextWindowExceeded as error:
            self.send_json(413, {"error": str(error), "token_report": error.report})
            return
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

        answer = agent_result["text"]
        complete = dialogue_mode and FINAL_MARKER in answer
        if complete:
            answer = answer.replace(FINAL_MARKER, "", 1).strip()
        elif finish_mode == "sequence" and finish_value in answer:
            answer = answer.split(finish_value, 1)[0].rstrip()
            complete = True
        answer = truncate_words(answer, max_words)
        selected_agent.replace_last_answer(answer)
        self.send_json(
            200,
            {
                "answer": answer,
                "complete": complete,
                "metrics": result_metrics(agent_result),
            },
        )

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
