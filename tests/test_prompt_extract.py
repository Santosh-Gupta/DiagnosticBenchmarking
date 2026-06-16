import unittest

from neurologybm.prompt_extract import extract_prompt_candidate


SAMPLE_CHALLENGE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record><metadata>
    <article article-type="case-report">
      <front>
        <journal-meta><journal-title-group><journal-title>Example Cases</journal-title></journal-title-group></journal-meta>
        <article-meta>
          <article-id pub-id-type="pmcid">PMC1</article-id>
          <article-id pub-id-type="doi">10.0000/example</article-id>
          <title-group><article-title>Photo Quiz</article-title></title-group>
          <permissions><license><license-p>https://creativecommons.org/licenses/by/4.0/</license-p></license></permissions>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Photo Quiz</title>
          <p>A patient has fever and headache. What is your diagnosis?</p>
        </sec>
        <sec>
          <title>Answer</title>
          <p>The diagnosis is example disease.</p>
        </sec>
      </body>
    </article>
  </metadata></record></GetRecord>
</OAI-PMH>
"""


class PromptExtractTests(unittest.TestCase):
    def test_extracts_prompt_and_answer_sections(self) -> None:
        row = extract_prompt_candidate(SAMPLE_CHALLENGE_XML)

        self.assertIn("What is your diagnosis", row["prompt_candidate"])
        self.assertIn("example disease", row["answer_rest_candidate"])
        self.assertEqual(row["extraction_confidence"], "high")
        self.assertEqual(row["likely_image_dependent"], "no")
        self.assertIs(row["ready_without_review"], True)


if __name__ == "__main__":
    unittest.main()

