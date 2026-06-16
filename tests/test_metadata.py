import unittest

from neurologybm.metadata import extract_article_metadata


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord>
    <record>
      <metadata>
        <article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="case-report">
          <front>
            <journal-meta>
              <journal-title-group>
                <journal-title>Example Neurology</journal-title>
              </journal-title-group>
              <publisher><publisher-name>Example Press</publisher-name></publisher>
            </journal-meta>
            <article-meta>
              <article-id pub-id-type="pmcid">PMC12345</article-id>
              <article-id pub-id-type="pmid">999</article-id>
              <article-id pub-id-type="doi">10.0000/example</article-id>
              <article-categories>
                <subj-group><subject>Case Report</subject></subj-group>
              </article-categories>
              <title-group><article-title>A hard neurologic case</article-title></title-group>
              <pub-date pub-type="epub"><year>2025</year><month>7</month><day>3</day></pub-date>
              <abstract><p>A patient presented with seizures.</p></abstract>
              <kwd-group><kwd>epilepsy</kwd><kwd>diagnosis</kwd></kwd-group>
              <permissions>
                <license>
                  <ali:license_ref xmlns:ali="http://www.niso.org/schemas/ali/1.0/">https://creativecommons.org/licenses/by/4.0/</ali:license_ref>
                  <license-p>This is open.</license-p>
                </license>
              </permissions>
            </article-meta>
          </front>
          <body><sec><title>Case presentation</title><p>Text</p></sec></body>
        </article>
      </metadata>
    </record>
  </GetRecord>
</OAI-PMH>
"""


class MetadataTests(unittest.TestCase):
    def test_extract_article_metadata_from_oai_jats(self) -> None:
        metadata = extract_article_metadata(SAMPLE_XML, license_profile="training")

        self.assertEqual(metadata["pmcid"], "PMC12345")
        self.assertEqual(metadata["pmid"], "999")
        self.assertEqual(metadata["doi"], "10.0000/example")
        self.assertEqual(metadata["title"], "A hard neurologic case")
        self.assertEqual(metadata["journal"], "Example Neurology")
        self.assertEqual(metadata["publication_date"], "2025-07-03")
        self.assertEqual(metadata["article_type"], "case-report")
        self.assertEqual(metadata["subjects"], ["Case Report"])
        self.assertEqual(metadata["keywords"], ["epilepsy", "diagnosis"])
        self.assertEqual(metadata["license_key"], "cc_by")
        self.assertIs(metadata["allowed_by_profile"], True)
        self.assertEqual(metadata["section_titles"], ["Case presentation"])


if __name__ == "__main__":
    unittest.main()
