import unittest

from neurologybm.comments import extract_visible_text_from_html


class CommentExtractionTests(unittest.TestCase):
    def test_extract_visible_text_omits_script_and_keeps_order(self) -> None:
        text = extract_visible_text_from_html(
            """
            <html><head><script>hidden()</script></head>
            <body>
              <h1>Case Discussion</h1>
              <p>First hypothesis.</p>
              <div>Second hypothesis<br>Next test.</div>
            </body></html>
            """
        )

        self.assertNotIn("hidden", text)
        self.assertLess(text.index("First hypothesis"), text.index("Second hypothesis"))
        self.assertIn("Next test", text)


if __name__ == "__main__":
    unittest.main()

