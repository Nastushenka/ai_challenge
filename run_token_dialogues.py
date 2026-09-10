"""Create three visible assignment-8 dialogue runs without expensive API calls."""

from agent import ConversationStore, estimate_messages_tokens
from token_scenarios import analyze_prepared_dialogues, prepared_dialogues


SESSION_IDS = (
    "task8-demo-short",
    "task8-demo-long",
    "task8-demo-overflow",
)


def create_visible_test_dialogues(model_key="deepseek-v4-pro", history_database=None):
    store = (
        ConversationStore(history_database)
        if history_database is not None
        else ConversationStore()
    )
    dialogues = prepared_dialogues(model_key)
    reports = analyze_prepared_dialogues(model_key)

    results = []
    for session_id, dialogue, scenario in zip(SESSION_IDS, dialogues, reports):
        store.clear(session_id)
        store.save(session_id, dialogue["messages"])
        for index, turn in enumerate(scenario["timeline"]):
            history_tokens = estimate_messages_tokens(
                dialogue["messages"][: index * 2]
            )
            estimated_input = turn["input_tokens"]
            overflow = turn["status"] == "overflow"
            token_report = {
                "current_request_tokens": max(0, estimated_input - history_tokens),
                "history_tokens": history_tokens,
                "instructions_tokens": 0,
                "estimated_input_tokens": estimated_input,
                "api_input_tokens": 0 if overflow else estimated_input,
                "response_tokens": 0 if overflow else turn["output_tokens"],
                "api_total_tokens": (
                    0 if overflow else estimated_input + turn["output_tokens"]
                ),
                "context_limit": scenario["context_limit"],
            }
            store.append_request_metrics(
                session_id,
                model_key,
                token_report,
                0.0 if overflow else turn["estimated_cost_usd"],
            )
        totals = store.usage_summary(session_id)
        results.append(
            {
                "session_id": session_id,
                "title": dialogue["title"],
                "messages": len(dialogue["messages"]),
                "turns": totals["turn_count"],
                "overflow": scenario["overflow"],
            }
        )
    return results


if __name__ == "__main__":
    for result in create_visible_test_dialogues():
        print(
            f"{result['session_id']}: {result['messages']} сообщений, "
            f"{result['turns']} ходов, overflow={result['overflow']}"
        )
