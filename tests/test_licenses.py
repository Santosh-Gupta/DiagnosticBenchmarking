import unittest

from neurologybm.licenses import is_allowed_by_profile, license_filter_query, license_key_from_href


class LicenseTests(unittest.TestCase):
    def test_training_profile_excludes_noncommercial_and_no_derivatives(self) -> None:
        query = license_filter_query("training")

        self.assertIn("cc0_license[filter]", query)
        self.assertIn("cc_by_license[filter]", query)
        self.assertIn("cc_by-sa_license[filter]", query)
        self.assertNotIn("cc_by-nc_license[filter]", query)
        self.assertNotIn("cc_by-nd_license[filter]", query)

    def test_license_href_mapping(self) -> None:
        self.assertEqual(license_key_from_href("http://creativecommons.org/licenses/by/4.0/"), "cc_by")
        self.assertEqual(
            license_key_from_href("https://creativecommons.org/licenses/by-nc-sa/4.0/"),
            "cc_by_nc_sa",
        )

    def test_profile_license_check(self) -> None:
        self.assertIs(is_allowed_by_profile("https://creativecommons.org/licenses/by/4.0/", "training"), True)
        self.assertIs(is_allowed_by_profile("https://creativecommons.org/licenses/by-nd/4.0/", "training"), False)
        self.assertIs(is_allowed_by_profile(None, "training"), None)


if __name__ == "__main__":
    unittest.main()
