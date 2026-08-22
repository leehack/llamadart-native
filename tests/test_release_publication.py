#!/usr/bin/env python3
"""Retry and collision tests for native release publication."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_publication import (  # noqa: E402
    Asset,
    ExistingRelease,
    PublicationError,
    build_desired_release,
    publish,
    reconcile_publication,
)


class ReleasePublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        assets = Path(self.temp.name)
        (assets / "assets.json").write_text("{}\n")
        (assets / "bundle.tar.gz").write_bytes(b"native bundle")
        self.desired = build_desired_release(
            tag="v0.2.0-1",
            native_commit="1" * 40,
            upstream_ref="v0.2.0",
            upstream_commit="2" * 40,
            prerelease=True,
            assets_dir=assets,
            workflow_run_url="https://github.com/example/repository/actions/runs/123",
            artifact_digest="sha256:" + "3" * 64,
        )

    def _release(
        self,
        *,
        draft: bool,
        assets: dict[str, Asset] | None = None,
        body: str | None = None,
    ) -> ExistingRelease:
        return ExistingRelease(
            draft=draft,
            prerelease=self.desired.prerelease,
            name=self.desired.name,
            body=self.desired.body if body is None else body,
            assets=self.desired.assets if assets is None else assets,
        )

    def test_retry_after_tag_creation_creates_draft_and_uploads_all(self) -> None:
        plan = reconcile_publication(
            self.desired, tag_target=self.desired.native_commit, release=None
        )
        self.assertFalse(plan.create_tag)
        self.assertTrue(plan.create_draft_release)
        self.assertEqual(set(plan.upload_assets), set(self.desired.assets))
        self.assertTrue(plan.publish_draft)

    def test_retry_resumes_only_matching_partial_draft(self) -> None:
        first_name = sorted(self.desired.assets)[0]
        release = self._release(
            draft=True,
            assets={first_name: self.desired.assets[first_name]},
        )
        plan = reconcile_publication(
            self.desired,
            tag_target=self.desired.native_commit,
            release=release,
        )
        self.assertEqual(
            plan.upload_assets,
            tuple(sorted(set(self.desired.assets) - {first_name})),
        )
        self.assertTrue(plan.publish_draft)

    def test_exact_published_retry_is_idempotent(self) -> None:
        plan = reconcile_publication(
            self.desired,
            tag_target=self.desired.native_commit,
            release=self._release(draft=False),
        )
        self.assertTrue(plan.complete)
        self.assertFalse(plan.create_tag)
        self.assertFalse(plan.create_draft_release)
        self.assertFalse(plan.upload_assets)
        self.assertFalse(plan.publish_draft)

    def test_conflicting_tag_fails_closed(self) -> None:
        with self.assertRaisesRegex(PublicationError, "immutable tag"):
            reconcile_publication(
                self.desired, tag_target="f" * 40, release=None
            )

    def test_conflicting_asset_or_correlation_fails_closed(self) -> None:
        first_name = sorted(self.desired.assets)[0]
        mismatched = dict(self.desired.assets)
        expected = mismatched[first_name]
        mismatched[first_name] = Asset(first_name, expected.size, "f" * 64)
        with self.assertRaisesRegex(PublicationError, "asset mismatch"):
            reconcile_publication(
                self.desired,
                tag_target=self.desired.native_commit,
                release=self._release(draft=True, assets=mismatched),
            )
        with self.assertRaisesRegex(PublicationError, "correlation mismatch"):
            reconcile_publication(
                self.desired,
                tag_target=self.desired.native_commit,
                release=self._release(draft=True, body="different transaction"),
            )

    def test_published_partial_release_fails_closed(self) -> None:
        with self.assertRaisesRegex(PublicationError, "incomplete"):
            reconcile_publication(
                self.desired,
                tag_target=self.desired.native_commit,
                release=self._release(draft=False, assets={}),
            )

    def test_release_without_tag_or_mismatched_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(PublicationError, "without its approved"):
            reconcile_publication(
                self.desired,
                tag_target=None,
                release=self._release(draft=True),
            )
        mismatched = ExistingRelease(
            draft=True,
            prerelease=False,
            name="different",
            body=self.desired.body,
            assets={},
        )
        with self.assertRaisesRegex(PublicationError, "name mismatch"):
            reconcile_publication(
                self.desired,
                tag_target=self.desired.native_commit,
                release=mismatched,
            )

    def test_new_workflow_transaction_has_distinct_correlation(self) -> None:
        rerun = build_desired_release(
            tag=self.desired.tag,
            native_commit=self.desired.native_commit,
            upstream_ref="v0.2.0",
            upstream_commit="2" * 40,
            prerelease=True,
            assets_dir=Path(self.temp.name),
            workflow_run_url="https://github.com/example/repository/actions/runs/456",
            artifact_digest="sha256:" + "4" * 64,
        )
        self.assertNotEqual(rerun.transaction_id, self.desired.transaction_id)
        with self.assertRaisesRegex(PublicationError, "correlation mismatch"):
            reconcile_publication(
                rerun,
                tag_target=rerun.native_commit,
                release=self._release(draft=True),
            )

    def test_transaction_binds_release_classification(self) -> None:
        stable_classification = build_desired_release(
            tag=self.desired.tag,
            native_commit=self.desired.native_commit,
            upstream_ref="v0.2.0",
            upstream_commit="2" * 40,
            prerelease=False,
            assets_dir=Path(self.temp.name),
            workflow_run_url="https://github.com/example/repository/actions/runs/123",
            artifact_digest="3" * 64,
        )
        self.assertNotEqual(
            stable_classification.transaction_id,
            self.desired.transaction_id,
        )

    def test_invalid_publication_correlation_is_rejected(self) -> None:
        base = {
            "tag": self.desired.tag,
            "native_commit": self.desired.native_commit,
            "upstream_ref": "v0.2.0",
            "upstream_commit": "2" * 40,
            "prerelease": True,
            "assets_dir": Path(self.temp.name),
            "workflow_run_url": "https://github.com/example/actions/runs/123",
            "artifact_digest": "3" * 64,
        }
        for digest in ("", "sha256:not-a-digest", "3" * 63):
            with self.subTest(digest=digest):
                with self.assertRaisesRegex(PublicationError, "SHA-256"):
                    build_desired_release(**{**base, "artifact_digest": digest})
        for run_url in (
            "",
            "http://github.com/example/repository/actions/runs/123",
            "https://github.com/example/repository/actions",
        ):
            with self.subTest(run_url=run_url):
                with self.assertRaisesRegex(PublicationError, "Actions run"):
                    build_desired_release(
                        **{**base, "workflow_run_url": run_url}
                    )

    def test_live_driver_is_draft_first_uploads_missing_and_publishes_last(self) -> None:
        state: dict[str, object] = {"tag": None, "release": None}
        commands: list[list[str]] = []

        def ensure_tag(_repository: str, desired) -> None:
            state["tag"] = desired.native_commit
            commands.append(["git", "push", "origin", f"refs/tags/{desired.tag}"])

        def release_json(_repository: str, _tag: str):
            return {} if state["release"] is not None else None

        def existing_release(_payload):
            return state["release"]

        def create(command, **_kwargs):
            commands.append(command)
            state["release"] = self._release(draft=True, assets={})
            return subprocess.CompletedProcess(command, 0, "", "")

        def run(command, **_kwargs):
            commands.append(command)
            release = state["release"]
            self.assertIsInstance(release, ExistingRelease)
            assert isinstance(release, ExistingRelease)
            if command[:3] == ["gh", "release", "upload"]:
                name = Path(command[4]).name
                assets = dict(release.assets)
                assets[name] = self.desired.assets[name]
                state["release"] = self._release(draft=True, assets=assets)
            elif command[:3] == ["gh", "release", "edit"]:
                state["release"] = self._release(draft=False, assets=dict(release.assets))
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch("release_publication._remote_tag_target", side_effect=lambda *_: state["tag"]),
            patch("release_publication._ensure_tag", side_effect=ensure_tag),
            patch("release_publication._release_json", side_effect=release_json),
            patch("release_publication._existing_release", side_effect=existing_release),
            patch("release_publication.subprocess.run", side_effect=create),
            patch("release_publication._run", side_effect=run),
        ):
            publish("example/repository", self.desired)

        create_index = next(
            index
            for index, command in enumerate(commands)
            if command[:3] == ["gh", "release", "create"]
        )
        upload_indexes = [
            index
            for index, command in enumerate(commands)
            if command[:3] == ["gh", "release", "upload"]
        ]
        publish_index = next(
            index
            for index, command in enumerate(commands)
            if command[:3] == ["gh", "release", "edit"]
        )
        self.assertLess(create_index, min(upload_indexes))
        self.assertLess(max(upload_indexes), publish_index)
        flattened = [part for command in commands for part in command]
        self.assertNotIn("--clobber", flattened)
        self.assertNotIn("--force", flattened)


if __name__ == "__main__":
    unittest.main()
