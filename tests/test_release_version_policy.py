#!/usr/bin/env python3
"""Release policy regression tests."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_version_policy import (  # noqa: E402
    PolicyError,
    manifest_native_tag,
    parse_native_tag,
    validate_automatic_upstream,
    validate_history,
    validate_pair,
    wrapper_tag_for,
)


class ReleaseVersionPolicyTest(unittest.TestCase):
    def test_stable_upstream_release(self) -> None:
        upstream, native = validate_pair("v0.2.0", "v0.2.0")
        self.assertEqual(upstream.channel, "stable")
        self.assertEqual(native.kind, "upstream")
        self.assertFalse(native.github_prerelease)

    def test_explicit_nightly_and_compact_wrapper_release(self) -> None:
        upstream, nightly = validate_pair("b10545", "b10545")
        _, rebuild = validate_pair("b10545", "b10545-1")
        self.assertTrue(nightly.github_prerelease)
        self.assertEqual(wrapper_tag_for(upstream), "b10545-1")
        self.assertEqual(rebuild.rebuild, 1)

    def test_legacy_nightly_wrapper_is_read_only_compatibility(self) -> None:
        legacy = parse_native_tag("b10356-llamadart.1")
        self.assertTrue(legacy.legacy)
        self.assertEqual(legacy.rebuild, 1)
        with self.assertRaisesRegex(PolicyError, "read-only compatibility"):
            validate_pair("b10356", legacy.tag)

    def test_automatic_discovery_rejects_nightly(self) -> None:
        stable, _ = validate_pair("v0.2.0", "v0.2.0")
        nightly, _ = validate_pair("b10545", "b10545")
        validate_automatic_upstream(stable)
        with self.assertRaisesRegex(PolicyError, "requires stable"):
            validate_automatic_upstream(nightly)

    def test_stable_wrapper_rebuild_is_monotonic(self) -> None:
        upstream, rebuild = validate_pair("v0.2.0", "v0.2.0-2")
        self.assertEqual(wrapper_tag_for(upstream), "v0.2.0-1")
        self.assertEqual(rebuild.core, upstream.core)
        self.assertTrue(rebuild.github_prerelease)
        validate_history(rebuild, ["v0.2.0", "v0.2.0-1"])
        validate_history(
            parse_native_tag("v0.2.0-10"),
            ["v0.2.0-9"],
        )
        validate_history(parse_native_tag("v0.2.1"), [rebuild.tag])

    def test_invalid_refs_and_mismatched_tags(self) -> None:
        with self.assertRaisesRegex(PolicyError, "PATCH-N"):
            parse_native_tag("not-a-release")

        invalid_pairs = (
            ("latest", "v0.2.0"),
            ("v0.2", "v0.2"),
            ("v0.2.0-rc.1", "v0.2.0-rc.1"),
            ("b010545", "b010545"),
            ("v0.2.0", "v0.2.0-llamadart.1"),
            ("v0.2.0", "v0.2.1-1"),
            ("v0.2.0", "v0.2.0-0"),
            ("b10545", "v0.2.0"),
            ("b10545", "b10545-0"),
        )
        for upstream, native in invalid_pairs:
            with self.subTest(upstream=upstream, native=native):
                with self.assertRaises(PolicyError):
                    validate_pair(upstream, native)

    def test_rollback_and_collision_are_rejected(self) -> None:
        with self.assertRaisesRegex(PolicyError, "collision"):
            validate_history(parse_native_tag("v0.2.0"), ["v0.2.0"])
        with self.assertRaisesRegex(PolicyError, "collision"):
            validate_history(parse_native_tag("v0.2.0-1"), ["v0.2.0-1"])
        with self.assertRaisesRegex(PolicyError, "collision"):
            validate_history(
                parse_native_tag("b10356-1"),
                ["b10356-llamadart.1"],
            )
        with self.assertRaisesRegex(PolicyError, "rollback"):
            validate_history(parse_native_tag("v0.1.9"), ["v0.2.0"])
        with self.assertRaisesRegex(PolicyError, "rollback"):
            validate_history(parse_native_tag("v0.2.0-3"), ["v0.2.1"])
        with self.assertRaisesRegex(PolicyError, "rollback"):
            validate_history(
                parse_native_tag("v0.2.0-1"),
                ["v0.2.0-2"],
            )
        with self.assertRaisesRegex(PolicyError, "rollback"):
            validate_history(
                parse_native_tag("b10356-1"),
                ["b10356-2"],
            )

    def test_explicit_historical_nightly_is_not_stable_rollback(self) -> None:
        candidate = parse_native_tag("b10356-2")
        validate_history(candidate, ["b10545", "v0.2.0", "b10356-llamadart.1"])

    def test_current_and_legacy_manifest_tags(self) -> None:
        self.assertEqual(
            manifest_native_tag(
                {"tag": "v0.2.0", "native_release_tag": "v0.2.0"}
            ),
            "v0.2.0",
        )
        self.assertEqual(manifest_native_tag({"tag": "b10514"}), "b10514")
        with self.assertRaises(PolicyError):
            manifest_native_tag(
                {"tag": "v0.2.0", "native_release_tag": "v0.2.1"}
            )


if __name__ == "__main__":
    unittest.main()
