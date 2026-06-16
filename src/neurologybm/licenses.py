"""PMC license filters and conservative reuse profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LicenseFilter:
    key: str
    pmc_filter: str
    allows_commercial: bool
    allows_derivatives: bool


LICENSE_FILTERS: dict[str, LicenseFilter] = {
    "cc0": LicenseFilter("cc0", "cc0_license[filter]", True, True),
    "cc_by": LicenseFilter("cc_by", "cc_by_license[filter]", True, True),
    "cc_by_sa": LicenseFilter("cc_by_sa", "cc_by-sa_license[filter]", True, True),
    "cc_by_nd": LicenseFilter("cc_by_nd", "cc_by-nd_license[filter]", True, False),
    "cc_by_nc": LicenseFilter("cc_by_nc", "cc_by-nc_license[filter]", False, True),
    "cc_by_nc_sa": LicenseFilter("cc_by_nc_sa", "cc_by-nc-sa_license[filter]", False, True),
    "cc_by_nc_nd": LicenseFilter("cc_by_nc_nd", "cc_by-nc-nd_license[filter]", False, False),
}


LICENSE_PROFILES: dict[str, tuple[str, ...]] = {
    # Default for model training data. It excludes NC and ND variants.
    "training": ("cc0", "cc_by", "cc_by_sa"),
    # PMC places CC BY-ND in the commercial-reuse grouping, but derivative work is
    # intentionally preserved as a separate opt-in profile.
    "commercial_reuse": ("cc0", "cc_by", "cc_by_sa", "cc_by_nd"),
    "noncommercial_training": (
        "cc0",
        "cc_by",
        "cc_by_sa",
        "cc_by_nc",
        "cc_by_nc_sa",
    ),
    "all_cc": tuple(LICENSE_FILTERS),
}


def profile_license_keys(profile: str) -> tuple[str, ...]:
    """Return license keys for a named profile."""

    try:
        return LICENSE_PROFILES[profile]
    except KeyError as exc:
        known = ", ".join(sorted(LICENSE_PROFILES))
        raise ValueError(f"Unknown license profile {profile!r}. Known profiles: {known}") from exc


def license_filter_query(profile: str) -> str:
    """Build the PMC ESearch license filter clause for a profile."""

    keys = profile_license_keys(profile)
    return "(" + " OR ".join(LICENSE_FILTERS[key].pmc_filter for key in keys) + ")"


def normalize_license_href(href: str | None) -> str | None:
    """Normalize a Creative Commons license URL or free-text fragment."""

    if not href:
        return None
    value = href.strip().lower()
    value = value.replace("http://", "https://")
    if value.endswith("/"):
        value = value[:-1]
    return value


def license_key_from_href(href: str | None) -> str | None:
    """Best-effort mapping from article license URLs to local license keys."""

    value = normalize_license_href(href)
    if not value:
        return None
    fragments = {
        "/publicdomain/zero/": "cc0",
        "/licenses/by/": "cc_by",
        "/licenses/by-sa/": "cc_by_sa",
        "/licenses/by-nd/": "cc_by_nd",
        "/licenses/by-nc/": "cc_by_nc",
        "/licenses/by-nc-sa/": "cc_by_nc_sa",
        "/licenses/by-nc-nd/": "cc_by_nc_nd",
    }
    for fragment, key in fragments.items():
        if fragment in value:
            return key
    return None


def is_allowed_by_profile(href: str | None, profile: str) -> bool | None:
    """Return whether a parsed article license fits a profile.

    None means the article did not expose a machine-readable CC URL that this
    helper recognizes. The search query should still be treated as the primary
    license gate.
    """

    key = license_key_from_href(href)
    if key is None:
        return None
    return key in profile_license_keys(profile)
