import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from htmlminer.formatting import normalize_bullet_list


class TestNormalizeBulletList(unittest.TestCase):
    def test_normalizes_bullet_characters(self):
        text = "\u2022 alpha\n\u2022 beta"
        expected = "- alpha\n- beta"
        self.assertEqual(normalize_bullet_list(text), expected)

    def test_normalizes_dash_lines(self):
        text = "- alpha\n- beta"
        expected = "- alpha\n- beta"
        self.assertEqual(normalize_bullet_list(text), expected)

    def test_normalizes_semicolon_list(self):
        text = "Items: alpha; beta; gamma"
        expected = "- alpha\n- beta\n- gamma"
        self.assertEqual(normalize_bullet_list(text), expected)

    def test_normalizes_inline_bullets(self):
        text = "\u2022 alpha \u2022 beta \u2022 gamma"
        expected = "- alpha\n- beta\n- gamma"
        self.assertEqual(normalize_bullet_list(text), expected)

    def test_passes_through_non_list(self):
        text = "Not mentioned."
        self.assertEqual(normalize_bullet_list(text), text)


if __name__ == "__main__":
    unittest.main()
