#!/usr/bin/env python3
"""Historical GitHub metadata regression coverage."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_version_policy import release_channel_from_metadata  # noqa: E402
from verify_historical_release_metadata import (  # noqa: E402
    FIXTURE,
    HISTORICAL_TARGETS,
    validate_historical_record,
)


class HistoricalReleaseMetadataTest(unittest.TestCase):
    def test_historical_false_prerelease_metadata_is_still_nightly(self) -> None:
        records = json.loads(FIXTURE.read_text())
        self.assertEqual(len(records), 2)
        for record in records:
            with self.subTest(tag=record["tag_name"]):
                self.assertFalse(record["prerelease"])
                self.assertEqual(
                    record["target_commitish"],
                    HISTORICAL_TARGETS[record["tag_name"]],
                )
                validate_historical_record(record)
                self.assertEqual(release_channel_from_metadata(record), "nightly")

    def test_new_metadata_channel_is_also_tag_driven(self) -> None:
        self.assertEqual(
            release_channel_from_metadata(
                {"tag_name": "b10545-1", "prerelease": True}
            ),
            "nightly",
        )
        self.assertEqual(
            release_channel_from_metadata(
                {"tag_name": "v0.2.0", "prerelease": False}
            ),
            "stable",
        )


if __name__ == "__main__":
    unittest.main()
