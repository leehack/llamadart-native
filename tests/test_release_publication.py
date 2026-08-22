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
    ExistingTag,
    PublicationError,
    _ensure_tag,
    _remote_tag,
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

    def _tag(
        self,
        *,
        target: str | None = None,
        transaction_id: str | None = None,
    ) -> ExistingTag:
        return ExistingTag(
            target or self.desired.native_commit,
            transaction_id or self.desired.transaction_id,
        )

    def test_same_transaction_retry_after_tag_creation_resumes(self) -> None:
        plan = reconcile_publication(
            self.desired, tag=self._tag(), release=None
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
            tag=self._tag(),
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
            tag=self._tag(),
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
                self.desired,
                tag=self._tag(target="f" * 40),
                release=None,
            )

    def test_tag_only_state_rejects_different_or_missing_transaction(self) -> None:
        for transaction_id in ("f" * 64, None):
            with self.subTest(transaction_id=transaction_id):
                tag = ExistingTag(self.desired.native_commit, transaction_id)
                with self.assertRaisesRegex(
                    PublicationError, "tag.*transaction mismatch"
                ):
                    reconcile_publication(self.desired, tag=tag, release=None)

    def test_conflicting_asset_or_correlation_fails_closed(self) -> None:
        first_name = sorted(self.desired.assets)[0]
        mismatched = dict(self.desired.assets)
        expected = mismatched[first_name]
        mismatched[first_name] = Asset(first_name, expected.size, "f" * 64)
        with self.assertRaisesRegex(PublicationError, "asset mismatch"):
            reconcile_publication(
                self.desired,
                tag=self._tag(),
                release=self._release(draft=True, assets=mismatched),
            )
        with self.assertRaisesRegex(PublicationError, "correlation mismatch"):
            reconcile_publication(
                self.desired,
                tag=self._tag(),
                release=self._release(draft=True, body="different transaction"),
            )

    def test_published_partial_release_fails_closed(self) -> None:
        with self.assertRaisesRegex(PublicationError, "incomplete"):
            reconcile_publication(
                self.desired,
                tag=self._tag(),
                release=self._release(draft=False, assets={}),
            )

    def test_release_without_tag_or_mismatched_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(PublicationError, "without its approved"):
            reconcile_publication(
                self.desired,
                tag=None,
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
                tag=self._tag(),
                release=mismatched,
            )

    def test_ambiguous_tag_push_accepts_only_exact_transaction_marker(self) -> None:
        exact = self._tag()
        commands: list[list[str]] = []

        def run(command, **_kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        ambiguous = subprocess.CompletedProcess(
            ["git", "push"],
            1,
            "",
            "connection closed after receiving objects",
        )
        with (
            patch(
                "release_publication._remote_tag",
                side_effect=[None, exact, exact],
            ),
            patch("release_publication._local_tag", return_value=None),
            patch("release_publication._run", side_effect=run),
            patch("release_publication.subprocess.run", return_value=ambiguous),
        ):
            _ensure_tag("example/repository", self.desired)

        create_tag = next(command for command in commands if "tag" in command)
        self.assertIn("-a", create_tag)
        self.assertIn(self.desired.transaction_id, create_tag[-1])

        for raced in (ExistingTag(self.desired.native_commit, "f" * 64), None):
            with self.subTest(raced=raced):
                with (
                    patch(
                        "release_publication._remote_tag",
                        side_effect=[None, raced],
                    ),
                    patch("release_publication._local_tag", return_value=None),
                    patch(
                        "release_publication._run",
                        return_value=subprocess.CompletedProcess([], 0, "", ""),
                    ),
                    patch(
                        "release_publication.subprocess.run",
                        return_value=ambiguous,
                    ),
                ):
                    with self.assertRaises(PublicationError):
                        _ensure_tag("example/repository", self.desired)

    def test_successful_push_must_be_observable_with_exact_marker(self) -> None:
        pushed = subprocess.CompletedProcess(["git", "push"], 0, "", "")
        with (
            patch(
                "release_publication._remote_tag",
                side_effect=[None, None],
            ),
            patch("release_publication._local_tag", return_value=None),
            patch(
                "release_publication._run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ),
            patch("release_publication.subprocess.run", return_value=pushed),
        ):
            with self.assertRaisesRegex(PublicationError, "not observable"):
                _ensure_tag("example/repository", self.desired)

    def test_remote_annotated_tag_exposes_transaction_marker(self) -> None:
        tag_object = "a" * 40
        responses = [
            {"object": {"type": "tag", "sha": tag_object}},
            {
                "message": (
                    "llamadart-native publication transaction: "
                    f"{self.desired.transaction_id}\n"
                ),
                "object": {
                    "type": "commit",
                    "sha": self.desired.native_commit,
                },
            },
        ]
        with patch(
            "release_publication._api_json_or_none",
            side_effect=responses,
        ):
            self.assertEqual(
                _remote_tag("example/repository", self.desired.tag),
                self._tag(),
            )

    def test_remote_lightweight_tag_is_unmarked(self) -> None:
        with patch(
            "release_publication._api_json_or_none",
            return_value={
                "object": {
                    "type": "commit",
                    "sha": self.desired.native_commit,
                }
            },
        ):
            self.assertEqual(
                _remote_tag("example/repository", self.desired.tag),
                ExistingTag(self.desired.native_commit, None),
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
        foreign_tag = ExistingTag(
            rerun.native_commit,
            self.desired.transaction_id,
        )
        with self.assertRaisesRegex(PublicationError, "transaction mismatch"):
            reconcile_publication(
                rerun,
                tag=foreign_tag,
                release=self._release(draft=True),
            )

        with (
            patch("release_publication._remote_tag", return_value=foreign_tag),
            patch("release_publication._release_json", return_value=None),
            patch("release_publication._ensure_tag") as ensure_tag,
            patch("release_publication.subprocess.run") as create_release,
        ):
            with self.assertRaisesRegex(PublicationError, "transaction mismatch"):
                publish("example/repository", rerun)
        ensure_tag.assert_not_called()
        create_release.assert_not_called()

    def test_driver_resumes_same_transaction_tag_only_state(self) -> None:
        state: dict[str, ExistingRelease | None] = {"release": None}

        def release_json(_repository: str, _tag: str):
            return {} if state["release"] is not None else None

        def existing_release(_payload):
            return state["release"]

        def create(command, **_kwargs):
            state["release"] = self._release(draft=True)
            return subprocess.CompletedProcess(command, 0, "", "")

        def run(command, **_kwargs):
            if command[:3] == ["gh", "release", "edit"]:
                state["release"] = self._release(draft=False)
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch("release_publication._remote_tag", return_value=self._tag()),
            patch("release_publication._release_json", side_effect=release_json),
            patch("release_publication._existing_release", side_effect=existing_release),
            patch("release_publication._ensure_tag") as ensure_tag,
            patch("release_publication.subprocess.run", side_effect=create),
            patch("release_publication._run", side_effect=run),
        ):
            publish("example/repository", self.desired)

        ensure_tag.assert_not_called()
        self.assertIsNotNone(state["release"])
        assert state["release"] is not None
        self.assertFalse(state["release"].draft)

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
            state["tag"] = ExistingTag(
                desired.native_commit,
                desired.transaction_id,
            )
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
            patch("release_publication._remote_tag", side_effect=lambda *_: state["tag"]),
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
