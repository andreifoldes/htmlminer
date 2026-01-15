import unittest
from unittest import mock

from htmlminer.agent import AgenticExtractor, MODEL_TIERS


class TestModelTier(unittest.TestCase):
    def test_invalid_model_tier_raises(self):
        with self.assertRaises(ValueError):
            AgenticExtractor(api_key="test-key", model_tier="ultra")

    def test_valid_model_tier_sets_models(self):
        tier = "cheap"
        with mock.patch("htmlminer.agent.dspy.LM") as mocked_lm, mock.patch(
            "htmlminer.agent.dspy.settings", new=mock.Mock()
        ):
            mocked_lm.return_value.history = []
            agent = AgenticExtractor(api_key="test-key", model_tier=tier)

        self.assertEqual(agent.model_id, MODEL_TIERS[tier]["model_id"])
        self.assertEqual(agent.dspy_model, MODEL_TIERS[tier]["dspy_model"])


if __name__ == "__main__":
    unittest.main()
