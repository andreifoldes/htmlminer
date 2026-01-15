import unittest
from unittest import mock
from pydantic import BaseModel

from htmlminer.firecrawl_agent import (
    FirecrawlAgentExtractor,
    SPARK_MODELS,
)


class TestSparkModels(unittest.TestCase):
    def test_invalid_spark_model_raises(self):
        with self.assertRaises(ValueError):
            FirecrawlAgentExtractor(api_key="test-key", spark_model="ultra")

    def test_valid_spark_models(self):
        for model_key, model_id in SPARK_MODELS.items():
            with mock.patch("htmlminer.firecrawl_agent.FirecrawlApp"):
                extractor = FirecrawlAgentExtractor(
                    api_key="test-key", spark_model=model_key
                )
            self.assertEqual(extractor.model_id, model_id)
            self.assertEqual(extractor.spark_model, model_key)


class TestBuildSchemaFromConfig(unittest.TestCase):
    def test_build_schema_creates_model_with_fields(self):
        with mock.patch("htmlminer.firecrawl_agent.FirecrawlApp"):
            extractor = FirecrawlAgentExtractor(api_key="test-key")

        features = [
            {"name": "Risk", "description": "Risk description"},
            {"name": "Goal", "description": "Goal description"},
        ]

        schema = extractor.build_schema_from_config(features)

        # Verify it's a Pydantic model
        self.assertTrue(issubclass(schema, BaseModel))

        # Verify fields exist
        field_names = set(schema.model_fields.keys())
        self.assertIn("Risk", field_names)
        self.assertIn("Goal", field_names)

    def test_build_prompt_includes_features(self):
        with mock.patch("htmlminer.firecrawl_agent.FirecrawlApp"):
            extractor = FirecrawlAgentExtractor(api_key="test-key")

        features = [
            {"name": "Risk", "description": "Risk description"},
        ]

        prompt = extractor.build_prompt_from_config("https://example.com", features)

        self.assertIn("https://example.com", prompt)
        self.assertIn("Risk", prompt)
        self.assertIn("Risk description", prompt)


class TestFirecrawlAgentRun(unittest.TestCase):
    def test_run_calls_agent_api(self):
        mock_app = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_result.data = {"Risk": "Some risk", "Goal": "Some goal"}
        mock_app.agent.return_value = mock_result

        with mock.patch(
            "htmlminer.firecrawl_agent.FirecrawlApp", return_value=mock_app
        ):
            extractor = FirecrawlAgentExtractor(api_key="test-key")

        features = [
            {"name": "Risk", "description": "Risk desc"},
            {"name": "Goal", "description": "Goal desc"},
        ]

        result = extractor.run(url="https://example.com", features=features)

        # Verify agent was called
        mock_app.agent.assert_called_once()

        # Verify result structure
        self.assertEqual(result["URL"], "https://example.com")
        self.assertEqual(result["Risk"], "Some risk")
        self.assertEqual(result["Goal"], "Some goal")


if __name__ == "__main__":
    unittest.main()
