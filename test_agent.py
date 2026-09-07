import json
import os
import unittest
from unittest.mock import patch

from agent import SimpleAgent


class FakeResponse:
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
                        "content": [{"type": "output_text", "text": "Готово"}],
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


if __name__ == "__main__":
    unittest.main()
