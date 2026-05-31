from __future__ import annotations


COUNT_PROFILES = {
    "standard": 50000,
    "max": 150000,
}


def profile_for_count(count: int) -> str:
    for profile, profile_count in COUNT_PROFILES.items():
        if count == profile_count:
            return profile
    return "custom"
