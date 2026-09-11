"""Prepared no-API comparison for assignment 9 context compression."""

from agent import (
    MODEL_OPTIONS,
    ConversationStore,
    estimate_messages_tokens,
    estimate_text_tokens,
)
from context_manager import build_compressed_context, summary_contains_fact


CONTROL_FACTS = ("Проект Аврора", "15 июня", "50 000 рублей")


def prepared_compression_dialogue():
    messages = [
        {
            "role": "user",
            "content": (
                "[Тест 9 · компрессия] Проект называется Проект Аврора. "
                "Мы готовим учебную презентацию "
                "для команды и подробно обсуждаем структуру, роли, порядок работы, "
                "риски, оформление и критерии проверки результата."
            ),
        },
        {
            "role": "assistant",
            "content": "Запомнила название проекта и буду учитывать его в следующих шагах.",
        },
        {
            "role": "user",
            "content": (
                "Срок сдачи — 15 июня. При планировании нужно подробно учитывать "
                "промежуточную проверку, репетицию, исправления и резерв времени."
            ),
        },
        {
            "role": "assistant",
            "content": "Срок 15 июня сохранён, этапы будут выстроены до этой даты.",
        },
        {
            "role": "user",
            "content": (
                "Бюджет — 50 000 рублей. В него входят материалы, дизайн, печать, "
                "тестирование и небольшой резерв на непредвиденные расходы."
            ),
        },
        {
            "role": "assistant",
            "content": "Бюджет 50 000 рублей принят как ограничение проекта.",
        },
    ]
    for turn in range(4, 16):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": (
                        f"Ход {turn}. Продолжи подробное обсуждение учебного проекта: "
                        "проверь задачи, ответственных, зависимости, риски и ожидаемый "
                        "результат этапа, не изменяя ранее согласованные факты."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f"Ход {turn}. План дополнен очередным этапом, ответственным, "
                        "проверяемым результатом и способом снизить выявленный риск."
                    ),
                },
            ]
        )
    return messages


def analyze_compression(recent_messages_limit=10, summary_batch_size=10):
    history = prepared_compression_dialogue()
    compressed = build_compressed_context(
        history,
        recent_messages_limit=recent_messages_limit,
        summary_batch_size=summary_batch_size,
    )
    before = estimate_messages_tokens(history)
    after = estimate_messages_tokens(compressed["messages"])
    checks = [summary_contains_fact(compressed["summary"], fact) for fact in CONTROL_FACTS]
    saved = max(0, before - after)
    return {
        "message_count": len(history),
        "recent_messages_limit": recent_messages_limit,
        "summary_batch_size": summary_batch_size,
        "summarized_message_count": compressed["summarized_message_count"],
        "raw_message_count": compressed["recent_message_count"],
        "tokens_without_compression": before,
        "tokens_with_compression": after,
        "tokens_saved": saved,
        "savings_percent": round(saved / before * 100, 1) if before else 0,
        "quality_without_compression": len(CONTROL_FACTS),
        "quality_with_compression": sum(checks),
        "quality_total": len(CONTROL_FACTS),
        "control_facts": [
            {"fact": fact, "preserved": preserved}
            for fact, preserved in zip(CONTROL_FACTS, checks)
        ],
        "test_question": "Назови название проекта, срок сдачи и бюджет.",
        "note": "Локальная проверка доступности контрольных фактов, без платного API-вызова.",
    }


def create_visible_compression_dialogue(
    model_key="deepseek-v4-pro", history_database=None
):
    """Save the prepared test, its summary, and comparable metrics for the UI."""
    session_id = "task9-demo-compression"
    store = (
        ConversationStore(history_database)
        if history_database is not None
        else ConversationStore()
    )
    history = prepared_compression_dialogue()
    report = analyze_compression()
    compressed = build_compressed_context(history)
    current_request_tokens = estimate_text_tokens(report["test_question"]) + 4
    history_tokens = estimate_messages_tokens(compressed["messages"])
    context_limit = MODEL_OPTIONS[model_key]["context_window"]

    store.clear(session_id)
    store.save(session_id, history)
    store.save_summary(
        session_id,
        compressed["summary"],
        compressed["summarized_message_count"],
        compressed["recent_messages_limit"],
    )
    store.append_request_metrics(
        session_id,
        model_key,
        {
            "current_request_tokens": current_request_tokens,
            "history_tokens": history_tokens,
            "instructions_tokens": 0,
            "estimated_input_tokens": history_tokens + current_request_tokens + 2,
            "api_input_tokens": 0,
            "response_tokens": 0,
            "api_total_tokens": 0,
            "context_limit": context_limit,
            "compression_enabled": True,
            "full_history_tokens": report["tokens_without_compression"],
            "summary_tokens": estimate_text_tokens(compressed["summary"]) + 4,
            "tokens_saved": report["tokens_saved"],
            "summarized_message_count": compressed["summarized_message_count"],
            "recent_messages_limit": compressed["recent_messages_limit"],
        },
        0.0,
    )
    return {"session_id": session_id, **report}


if __name__ == "__main__":
    result = create_visible_compression_dialogue()
    print(
        f"{result['session_id']}: {result['tokens_without_compression']} -> "
        f"{result['tokens_with_compression']} токенов, "
        f"экономия {result['savings_percent']}%"
    )
