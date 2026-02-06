import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


class TestDeepMindIntegration(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        "GEMINI_API_KEY or GOOGLE_API_KEY required for live integration test.",
    )
    def test_deepmind_url_workflow(self):
        try:
            from htmlminer.graph_workflow import run_htmlminer_workflow
        except ModuleNotFoundError as exc:
            self.skipTest(f"Dependency missing for integration test: {exc}")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        features = [
            {
                "name": "Overview",
                "description": "A concise overview of DeepMind's mission and research focus.",
                "synthesis_topic": "DeepMind's mission and research focus",
                "length": "1 short paragraph (2-4 sentences)",
            }
        ]

        final_state = run_htmlminer_workflow(
            url="https://deepmind.google",
            features=features,
            api_key=api_key,
            engine="trafilatura",
            smart_mode=False,
            limit=1,
            model_tier="cheap",
            use_langextract=False,
        )

        results = final_state.get("results", {})
        self.assertEqual(results.get("URL"), "https://deepmind.google")
        summary = results.get("Overview", "")
        self.assertIsInstance(summary, str)
        self.assertTrue(summary.strip())


if __name__ == "__main__":
    unittest.main()
