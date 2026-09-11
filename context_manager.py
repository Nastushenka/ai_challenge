"""Incremental dialogue-history compression for the persistent LLM agent."""

import re


SUMMARY_MESSAGE_PREFIX = (
    "Краткое содержание более ранней части диалога. "
    "Используй эти факты как контекст и не придумывай отсутствующие детали:\n"
)


def _compact_message(message, max_chars=70):
    """Keep the useful beginning of a message and remove verbose formatting."""
    content = " ".join(str(message.get("content", "")).split())
    if len(content) > max_chars:
        boundary = max(content.rfind(". ", 0, max_chars), content.rfind("; ", 0, max_chars))
        content = content[: boundary + 1 if boundary >= 60 else max_chars].rstrip()
        content += "…"
    role = "Пользователь" if message.get("role") == "user" else "Ассистент"
    return f"- {role}: {content}"


def extend_summary(previous_summary, messages):
    """Append a compact extractive summary for one completed message batch."""
    additions = [_compact_message(message) for message in messages if message.get("content")]
    return "\n".join(part for part in (previous_summary.strip(), *additions) if part)


def build_compressed_context(
    history,
    previous_summary="",
    summarized_message_count=0,
    recent_messages_limit=10,
    summary_batch_size=10,
):
    """Return summary + recent messages and the updated persistent summary state."""
    recent_messages_limit = max(1, int(recent_messages_limit))
    summary_batch_size = max(1, int(summary_batch_size))
    eligible_count = max(0, len(history) - recent_messages_limit)
    target_count = eligible_count - eligible_count % summary_batch_size

    summary = previous_summary or ""
    summarized_count = max(0, int(summarized_message_count or 0))
    if summarized_count > target_count:
        summary = ""
        summarized_count = 0

    while summarized_count < target_count:
        batch_end = min(summarized_count + summary_batch_size, target_count)
        summary = extend_summary(summary, history[summarized_count:batch_end])
        summarized_count = batch_end

    context = []
    if summary:
        context.append({"role": "system", "content": SUMMARY_MESSAGE_PREFIX + summary})
    context.extend(history[summarized_count:])
    return {
        "messages": context,
        "summary": summary,
        "summarized_message_count": summarized_count,
        "recent_message_count": len(history) - summarized_count,
        "recent_messages_limit": recent_messages_limit,
        "summary_batch_size": summary_batch_size,
    }


def summary_contains_fact(summary, fact):
    """Case-insensitive quality check used by the prepared demonstration."""
    normalize = lambda value: re.sub(r"\s+", " ", str(value).strip().lower())
    return normalize(fact) in normalize(summary)
