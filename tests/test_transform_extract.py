import unittest

from neurologybm.transform_extract import extract_transformed_challenge


SAMPLE_CASE_REPORT_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record><metadata>
    <article article-type="case-report">
      <front>
        <journal-meta><journal-title-group><journal-title>Example Neurology</journal-title></journal-title-group></journal-meta>
        <article-meta>
          <article-id pub-id-type="pmcid">PMC2</article-id>
          <article-id pub-id-type="doi">10.0000/example2</article-id>
          <title-group><article-title>Autoimmune encephalitis presenting as psychosis</article-title></title-group>
          <permissions><license><license-p>https://creativecommons.org/licenses/by/4.0/</license-p></license></permissions>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Case presentation</title>
          <p>A 22-year-old woman presented with two weeks of paranoia, insomnia, and new generalized seizures.</p>
          <p>Examination showed fluctuating attention and orofacial dyskinesias.</p>
          <p>Initial toxicology screening was negative. Cerebrospinal fluid showed mild lymphocytic pleocytosis, and brain magnetic resonance imaging did not show a mass lesion or acute infarct.</p>
          <p>The treating team considered primary psychiatric disease, viral encephalitis, autoimmune encephalitis, and medication-related causes because the presentation crossed psychiatric and neurologic boundaries.</p>
          <p>She was diagnosed with autoimmune encephalitis after antibody testing.</p>
        </sec>
        <sec>
          <title>Discussion</title>
          <p>The final diagnosis was anti-NMDA receptor encephalitis. Early immunotherapy is recommended.</p>
        </sec>
      </body>
    </article>
  </metadata></record></GetRecord>
</OAI-PMH>
"""


class TransformExtractTests(unittest.TestCase):
    def test_transforms_case_report_without_answer_leakage(self) -> None:
        row = extract_transformed_challenge(SAMPLE_CASE_REPORT_XML)

        self.assertEqual(row["status"], "definitive_human_text_transformed")
        self.assertIn("22-year-old woman", row["challenge_prompt"])
        self.assertNotIn("antibody testing", row["challenge_prompt"])
        self.assertIn("anti-NMDA receptor encephalitis", row["answer_rest"])


if __name__ == "__main__":
    unittest.main()
