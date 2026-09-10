import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import (
    MODEL_OPTIONS,
    ContextWindowExceeded,
    ConversationStore,
    SimpleAgent,
    estimate_messages_tokens,
)
from run_token_dialogues import SESSION_IDS, create_visible_test_dialogues
from token_scenarios import analyze_prepared_dialogues, prepared_dialogues


class FakeResponse:
    def __init__(self, text="Готово"):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": self.text}],
                    }
                ],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 0},
                },
            }
        ).encode()


class SimpleAgentTest(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch("agent.urlopen", return_value=FakeResponse())
    def test_agent_sends_request_and_returns_answer(self, mocked_urlopen):
        agent = SimpleAgent("deepseek-v4-pro")

        result = agent.run("Проверка", "Ответь кратко", temperature=0)

        self.assertEqual(result["text"], "Готово")
        self.assertEqual(result["usage"]["total_tokens"], 6)
        self.assertGreater(result["token_report"]["current_request_tokens"], 0)
        self.assertEqual(result["token_report"]["history_tokens"], 0)
        self.assertEqual(result["token_report"]["api_input_tokens"], 4)
        self.assertEqual(result["token_report"]["response_tokens"], 2)
        request = mocked_urlopen.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["input"], "Проверка")
        self.assertEqual(body["instructions"], "Ответь кратко")
        self.assertEqual(body["temperature"], 0)

    @patch.dict(os.environ, {"LLM_API_KEY": "test-key"})
    @patch(
        "agent.urlopen",
        side_effect=[FakeResponse("Запомнила: вас зовут Настя"), FakeResponse("Настя")],
    )
    def test_history_survives_agent_restart(self, mocked_urlopen):
        with tempfile.TemporaryDirectory() as temporary_directory:
            history_database = Path(temporary_directory) / "history.db"
            first_agent = SimpleAgent(
                "deepseek-v4-pro",
                session_id="test-session",
                history_database=history_database,
            )
            first_agent.chat("Меня зовут Настя")

            restarted_agent = SimpleAgent(
                "deepseek-v4-pro",
                session_id="test-session",
                history_database=history_database,
            )
            restarted_agent.chat("Как меня зовут?")

            second_request = mocked_urlopen.call_args_list[1].args[0]
            second_input = json.loads(second_request.data)["input"]
            self.assertEqual(second_input[0]["content"], "Меня зовут Настя")
            self.assertEqual(second_input[1]["role"], "assistant")
            self.assertEqual(second_input[2]["content"], "Как меня зовут?")
            conversations = restarted_agent.store.list()
            self.assertEqual(conversations[0]["id"], "test-session")
            self.assertEqual(conversations[0]["message_count"], 4)
            self.assertEqual(conversations[0]["title"], "Меня зовут Настя")
            totals = restarted_agent.store.usage_summary("test-session")
            self.assertEqual(totals["turn_count"], 2)
            self.assertEqual(totals["cumulative_api_tokens"], 12)
            self.assertGreater(totals["turns"][1]["history_tokens"], 0)

    def test_store_keeps_complete_history_without_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ConversationStore(Path(temporary_directory) / "history.db")
            messages = [
                {
                    "role": "user" if index % 2 == 0 else "assistant",
                    "content": f"Сообщение {index}",
                }
                for index in range(60)
            ]

            store.save("long-session", messages)

            self.assertEqual(store.get("long-session"), messages)
            self.assertEqual(store.list()[0]["message_count"], 60)

    def test_legacy_json_history_is_migrated_to_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            legacy_file = directory / "history.json"
            database = directory / "history.db"
            legacy_file.write_text(
                json.dumps(
                    {
                        "legacy-session": {
                            "messages": [
                                {"role": "user", "content": "Старый вопрос"},
                                {"role": "assistant", "content": "Старый ответ"},
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            store = ConversationStore(database, legacy_json_path=legacy_file)

            self.assertEqual(len(store.get("legacy-session")), 2)

    @patch.dict(
        os.environ,
        {"LLM_API_KEY": "test-key", "DEEPSEEK_CONTEXT_WINDOW": "10"},
    )
    @patch("agent.urlopen")
    def test_context_overflow_is_stopped_before_api_call(self, mocked_urlopen):
        agent = SimpleAgent("deepseek-v4-pro")

        with self.assertRaises(ContextWindowExceeded) as raised:
            agent.run("Очень длинный запрос, который точно не помещается в десять токенов")

        self.assertEqual(raised.exception.report["status"], "overflow")
        mocked_urlopen.assert_not_called()

    def test_prepared_dialogues_cover_short_long_and_overflow(self):
        scenarios = analyze_prepared_dialogues("qwen3-8b")

        self.assertEqual([item["id"] for item in scenarios], ["short", "long", "overflow"])
        self.assertFalse(scenarios[0]["overflow"])
        self.assertFalse(scenarios[1]["overflow"])
        self.assertTrue(scenarios[2]["overflow"])
        long_inputs = [turn["input_tokens"] for turn in scenarios[1]["timeline"]]
        long_costs = [turn["cumulative_cost_usd"] for turn in scenarios[1]["timeline"]]
        self.assertEqual(long_inputs, sorted(long_inputs))
        self.assertEqual(long_costs, sorted(long_costs))

    def test_visible_dialogues_and_cards_use_identical_messages(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "history.db"
            create_visible_test_dialogues("qwen3-8b", database)
            store = ConversationStore(database)
            scenarios = analyze_prepared_dialogues("qwen3-8b")

            for session_id, scenario in zip(SESSION_IDS, scenarios):
                self.assertEqual(
                    estimate_messages_tokens(store.get(session_id)),
                    scenario["dialogue_tokens"],
                )

    def test_deepseek_context_window_matches_current_model(self):
        self.assertEqual(MODEL_OPTIONS["deepseek-v4-pro"]["context_window"], 1_000_000)


if __name__ == "__main__":
    unittest.main()
