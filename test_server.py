import unittest
from unittest.mock import patch

import server
from compression_scenarios import score_answer_facts


class FakeAgent:
    def __init__(self, model_key="deepseek-v4-pro"):
        self.model_key = model_key

    def run(self, user_request, instructions=None, temperature=None):
        text = user_request if isinstance(user_request, str) else "Диалог"
        return {
            "text": f"Ответ: {text[:20]}",
            "elapsed_seconds": 0.1,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            "cost_usd": 0.00001,
            "pricing_note": "Тестовый расчёт",
        }


class ComparisonMetricsTest(unittest.TestCase):
    def test_compression_quality_score_finds_all_control_facts(self):
        score = score_answer_facts(
            "Проект Аврора\nСрок: 15 июня\nБюджет: 50 000 рублей"
        )

        self.assertEqual(score["found"], 3)
        self.assertEqual(score["total"], 3)

    @patch("server.SimpleAgent", FakeAgent)
    def test_four_approaches_include_metrics(self):
        result = server.compare_solutions("Задача", None)

        self.assertEqual(len(result["solutions"]), 4)
        self.assertTrue(all("metrics" in item for item in result["solutions"]))
        self.assertEqual(result["analysis_metrics"]["total_tokens"], 15)

    @patch("server.SimpleAgent", FakeAgent)
    def test_temperature_comparison_includes_metrics(self):
        result = server.compare_temperatures("Задача", None)

        self.assertEqual(len(result["solutions"]), 3)
        self.assertTrue(all("metrics" in item for item in result["solutions"]))

    @patch("server.SimpleAgent", FakeAgent)
    def test_model_comparison_includes_metrics(self):
        result = server.compare_models("Задача", None)

        self.assertEqual(len(result["solutions"]), 3)
        self.assertEqual(result["analysis_metrics"]["total_tokens"], 15)

    @patch("server.SimpleAgent", FakeAgent)
    def test_live_compression_quality_compares_two_real_request_shapes(self):
        result = server.run_compression_quality_test()

        self.assertEqual(result["model_label"], "DeepSeek V4 Pro")
        self.assertIn("full", result)
        self.assertIn("compressed", result)
        self.assertGreater(
            result["full"]["estimated_context_tokens"],
            result["compressed"]["estimated_context_tokens"],
        )


if __name__ == "__main__":
    unittest.main()
