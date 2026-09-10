"""Prepared dialogue scenarios for assignment 8 token-limit experiments."""

from agent import MODEL_OPTIONS, estimate_messages_tokens, estimate_text_tokens


def _estimated_cost(model_key, input_tokens, output_tokens):
    if model_key == "deepseek-v4-pro":
        input_rate, output_rate = 0.66, 1.98
    else:
        pricing = MODEL_OPTIONS[model_key]["pricing"]
        input_rate, output_rate = pricing["input"], pricing["output"]
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _dialogue_report(dialogue, model_key):
    limit = MODEL_OPTIONS[model_key]["context_window"]
    history = []
    cumulative_cost = 0.0
    timeline = []
    overflow = False
    messages = dialogue["messages"]
    for index in range(0, len(messages), 2):
        user_message = messages[index]
        assistant_message = messages[index + 1] if index + 1 < len(messages) else None
        input_tokens = estimate_messages_tokens(history + [user_message])
        output_tokens = (
            estimate_text_tokens(assistant_message["content"]) + 4
            if assistant_message
            else 0
        )
        turn_cost = _estimated_cost(model_key, input_tokens, output_tokens)
        cumulative_cost += turn_cost
        turn_overflow = input_tokens > limit
        overflow = overflow or turn_overflow
        timeline.append(
            {
                "turn": index // 2 + 1,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_cost_usd": round(turn_cost, 8),
                "cumulative_cost_usd": round(cumulative_cost, 8),
                "context_usage_percent": round(input_tokens / limit * 100, 2),
                "status": "overflow" if turn_overflow else "ok",
            }
        )
        history.append(user_message)
        if assistant_message:
            history.append(assistant_message)
    return {
        "id": dialogue["id"],
        "title": dialogue["title"],
        "description": dialogue["description"],
        "message_count": len(messages),
        "dialogue_tokens": estimate_messages_tokens(messages),
        "context_limit": limit,
        "overflow": overflow,
        "expected_behavior": (
            "Агент остановит запрос до API и вернёт HTTP 413 с отчётом о переполнении."
            if overflow
            else "Диалог помещается в контекст; вход и стоимость растут на каждом ходе."
        ),
        "timeline": timeline,
    }


def prepared_dialogues(model_key="deepseek-v4-pro"):
    """Return short, long, and deliberately overflowing dialogue fixtures."""
    limit = MODEL_OPTIONS[model_key]["context_window"]
    short = {
        "id": "short",
        "title": "Короткий диалог",
        "description": "Два коротких обмена: проверка базового подсчёта.",
        "messages": [
            {
                "role": "user",
                "content": "[Тест 1 · короткий] Кодовое слово — Лазурь. Запомни его.",
            },
            {"role": "assistant", "content": "Запомнил: кодовое слово — Лазурь."},
            {"role": "user", "content": "Какое кодовое слово?"},
            {
                "role": "assistant",
                "content": "Кодовое слово — Лазурь. Контекст короткого диалога сохранён.",
            },
        ],
    }
    long_messages = []
    for turn in range(1, 31):
        long_messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        ("[Тест 2 · длинный] " if turn == 1 else "")
                        + f"Ход {turn}. Продолжай разрабатывать план учебного проекта. "
                        "Учитывай цели, ограничения, сроки, риски и результаты всех "
                        "предыдущих сообщений. Добавь одно новое обоснованное решение."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Ход {turn}: добавлено решение с проверкой целей, сроков и рисков. "
                        "Предыдущие договорённости сохранены и использованы в плане."
                    ),
                },
            ]
        )
    long_dialogue = {
        "id": "long",
        "title": "Длинный диалог",
        "description": "30 обменов: видно накопление входных токенов и стоимости.",
        "messages": long_messages,
    }
    overflow_text = (
        "[Тест 3 · переполнение] "
        + "важный контекст " * (limit // 4)
    )
    overflow_dialogue = {
        "id": "overflow",
        "title": "Диалог сверх лимита",
        "description": "Контекст намеренно больше окна выбранной модели.",
        "messages": [{"role": "user", "content": overflow_text}],
    }
    return [short, long_dialogue, overflow_dialogue]


def analyze_prepared_dialogues(model_key="deepseek-v4-pro"):
    if model_key not in MODEL_OPTIONS:
        raise ValueError("Неизвестная модель.")
    return [_dialogue_report(dialogue, model_key) for dialogue in prepared_dialogues(model_key)]
