import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import ConversationStore, SimpleAgent


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


if __name__ == "__main__":
    unittest.main()
