#!/usr/bin/env python3
"""Verify immutable historical nightly metadata and tag-based classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_version_policy import (  # noqa: E402
    PolicyError,
    release_channel_from_metadata,
)


FIXTURE = ROOT / "tests" / "fixtures" / "historical_release_metadata.json"
HISTORICAL_TAGS = ("b10545", "b10356-llamadart.1")
HISTORICAL_TARGETS = {
    "b10545": "71cf5aa5341439f6b1b2177d658ef040db28745d",
    "b10356-llamadart.1": "e5af4a73ea66c7a8314c219086c98717d847fccf",
}


def validate_historical_record(record: Mapping[str, Any]) -> None:
    tag = record.get("tag_name")
    if tag not in HISTORICAL_TAGS:
        raise PolicyError(f"unexpected historical release fixture {tag!r}")
    if record.get("draft") is not False or record.get("prerelease") is not False:
        raise PolicyError(
            f"historical release {tag!r} must retain observed draft=false and "
            "prerelease=false metadata"
        )
    if record.get("target_commitish") != HISTORICAL_TARGETS[tag]:
        raise PolicyError(
            f"historical release {tag!r} target changed from immutable commit "
            f"{HISTORICAL_TARGETS[tag]}"
        )
    if release_channel_from_metadata(record) != "nightly":
        raise PolicyError(
            f"historical release {tag!r} must be classified as nightly from tag grammar"
        )


def _live_record(repository: str, tag: str) -> Mapping[str, Any]:
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PolicyError(f"unable to read live historical release {tag!r}: {detail}")
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--repository", default="leehack/llamadart-native")
    args = parser.parse_args()

    try:
        records = json.loads(args.fixture.read_text())
        if {record.get("tag_name") for record in records} != set(HISTORICAL_TAGS):
            raise PolicyError("historical fixture must cover both immutable nightly tags")
        for record in records:
            validate_historical_record(record)
        if args.live:
            for tag in HISTORICAL_TAGS:
                validate_historical_record(_live_record(args.repository, tag))
    except (OSError, json.JSONDecodeError, PolicyError) as error:
        parser.error(str(error))

    mode = "fixture and live" if args.live else "fixture"
    print(f"Historical release metadata verified ({mode}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
