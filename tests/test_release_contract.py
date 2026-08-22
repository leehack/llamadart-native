from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from release_contract import (  # noqa: E402
    ContractError,
    RUNTIME_BUNDLES,
    build_discovery_report,
    build_release_result,
    validate_dispatch,
)
from release_publication import build_desired_release  # noqa: E402


COMMIT = "1" * 40
NATIVE_COMMIT = "2" * 40
HEAD_COMMIT = "3" * 40


class DispatchContractTests(unittest.TestCase):
    def test_exact_approved_publication_contract(self) -> None:
        contract = validate_dispatch(
            upstream_ref="v0.2.0",
            upstream_commit=COMMIT.upper(),
            native_release_tag="v0.2.0-1",
            smoke_policy="required",
            correlation_id="llamadart/400/run-123",
            publish_release=True,
        )
        self.assertEqual(COMMIT, contract["llama_cpp_commit"])
        self.assertEqual("v0.2.0-1", contract["native_release_tag"])
        self.assertEqual(contract["native_release_tag"], contract["tag"])

    def test_publish_rejects_moving_or_ambiguous_inputs(self) -> None:
        cases = (
            {"upstream_ref": "latest"},
            {"upstream_commit": "1" * 39},
            {"native_release_tag": "b10545-llamadart.2", "upstream_ref": "b10545"},
            {"smoke_policy": "skip"},
            {"correlation_id": "unsafe value"},
        )
        base = {
            "upstream_ref": "v0.2.0",
            "upstream_commit": COMMIT,
            "native_release_tag": "v0.2.0",
            "smoke_policy": "required",
            "correlation_id": "central-123",
            "publish_release": True,
        }
        for update in cases:
            with self.subTest(update=update), self.assertRaises(ContractError):
                validate_dispatch(**(base | update))

    def test_build_only_may_explicitly_skip_smoke(self) -> None:
        contract = validate_dispatch(
            upstream_ref="b10545",
            upstream_commit=COMMIT,
            native_release_tag="b10545-1",
            smoke_policy="skip",
            correlation_id="preparation-1",
            publish_release=False,
        )
        self.assertEqual("nightly", contract["upstream_channel"])


class DiscoveryContractTests(unittest.TestCase):
    def test_candidate_noop_and_incompatibility_are_machine_readable(self) -> None:
        for status in ("candidate", "noop", "incompatible"):
            with self.subTest(status=status):
                report = build_discovery_report(
                    status=status,
                    upstream_ref="v0.2.0" if status != "incompatible" else "future",
                    upstream_commit=COMMIT,
                    native_head_commit=HEAD_COMMIT,
                    workflow_run_id="123",
                    workflow_run_url="https://github.com/leehack/llamadart-native/actions/runs/123",
                    message="fixture",
                )
                self.assertEqual(status, report["status"])
                self.assertEqual(status == "candidate", report["candidate"])
                self.assertEqual(COMMIT, report["llama_cpp_commit"])
                self.assertEqual(report["native_release_tag"], report["tag"])


class ReleaseResultTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, dict[str, object]]:
        assets = root / "release_assets"
        assets.mkdir()
        tag = "v0.2.0"
        names = {
            f"llamadart-native-{bundle}-{tag}.tar.gz" for bundle in RUNTIME_BUNDLES
        }
        names.update(
            {
                f"llamadart-native-apple-xcframework-{tag}.zip",
                f"llamadart-native-headers-{tag}.tar.gz",
            }
        )
        artifacts = []
        checksums = []
        for name in sorted(names):
            path = assets / name
            path.write_bytes(f"fixture:{name}".encode())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            checksums.append(f"{digest}  {name}")
            artifacts.append(
                {"file": name, "sha256": digest, "size": path.stat().st_size}
            )
        (assets / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
        (assets / "assets.json").write_text(
            json.dumps(
                {
                    "tag": tag,
                    "native_release_tag": tag,
                    "llama_cpp_tag": tag,
                    "llama_cpp_commit": COMMIT,
                    "native_commit": NATIVE_COMMIT,
                    "correlation_id": "central-123",
                    "smoke_policy": "required",
                    "smoke_conclusion": "passed",
                    "artifacts": artifacts,
                }
            )
        )
        remote_names = names | {"assets.json", "SHA256SUMS"}
        release = {
            "id": 42,
            "html_url": "https://github.com/leehack/llamadart-native/releases/tag/v0.2.0",
            "tag_name": tag,
            "draft": False,
            "prerelease": False,
            "published_at": "2026-08-22T00:00:00Z",
            "body": (
                "publication transaction: `" + "a" * 64 + "`\n"
                "orchestrator correlation: `central-123`\n"
                "workflow run: https://github.com/leehack/llamadart-native/actions/runs/123\n"
                f"workflow head SHA: `{HEAD_COMMIT}`\n"
            ),
            "assets": [
                {
                    "id": index,
                    "name": name,
                    "browser_download_url": f"https://example.invalid/{name}",
                    "size": (assets / name).stat().st_size,
                    "digest": f"sha256:{hashlib.sha256((assets / name).read_bytes()).hexdigest()}",
                }
                for index, name in enumerate(sorted(remote_names), 1)
            ],
        }
        release["body"] = build_desired_release(
            tag=tag,
            native_commit=NATIVE_COMMIT,
            upstream_ref=tag,
            upstream_commit=COMMIT,
            prerelease=False,
            assets_dir=assets,
            workflow_run_url="https://github.com/leehack/llamadart-native/actions/runs/123",
            artifact_digest="a" * 64,
            correlation_id="central-123",
            smoke_policy="required",
            smoke_conclusion="passed",
            workflow_head_sha=HEAD_COMMIT,
        ).body
        return assets, release

    def _result(self, assets: Path, release: dict[str, object]) -> dict[str, object]:
        return build_release_result(
            assets_dir=assets,
            release_metadata=release,
            native_release_tag="v0.2.0",
            upstream_ref="v0.2.0",
            upstream_commit=COMMIT,
            native_commit=NATIVE_COMMIT,
            correlation_id="central-123",
            smoke_policy="required",
            smoke_conclusion="passed",
            publish_release=True,
            workflow_repository="leehack/llamadart-native",
            workflow_run_id="123",
            workflow_run_attempt="1",
            workflow_run_url="https://github.com/leehack/llamadart-native/actions/runs/123",
            workflow_head_sha=HEAD_COMMIT,
            publication_artifact_id="456",
            publication_artifact_url="https://github.com/example/artifacts/456",
            publication_artifact_digest="a" * 64,
        )

    def test_result_returns_exact_provenance_coverage_digests_and_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            result = self._result(assets, release)
        self.assertEqual("published", result["status"])
        self.assertEqual("v0.2.0", result["native_release_tag"])
        self.assertEqual(result["native_release_tag"], result["tag"])
        self.assertTrue(result["bundle_coverage"]["complete"])
        self.assertEqual("passed", result["smoke"]["conclusion"])
        self.assertEqual(42, result["release"]["id"])

    def test_result_rejects_digest_or_bundle_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            target = next(assets.glob("*.tar.gz"))
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(ContractError, "digest mismatch"):
                self._result(assets, release)

    def test_result_rejects_non_object_manifest_with_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            (assets / "assets.json").write_text("[]\n")
            with self.assertRaisesRegex(
                ContractError, "assets.json must contain a JSON object"
            ):
                self._result(assets, release)

    def test_result_rejects_malformed_manifest_asset_entries(self) -> None:
        for mutation, message in (
            (lambda artifacts: artifacts.append(None), "must be a JSON object"),
            (lambda artifacts: artifacts[0].pop("file"), "must have a file name"),
            (lambda artifacts: artifacts[0].update(size=None), "integer size"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                assets, release = self._fixture(Path(directory))
                manifest = json.loads((assets / "assets.json").read_text())
                mutation(manifest["artifacts"])
                (assets / "assets.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ContractError, message):
                    self._result(assets, release)

    def test_result_rejects_malformed_github_asset_entries(self) -> None:
        for mutation, message in (
            (lambda items: items.append(None), "must be a JSON object"),
            (lambda items: items[0].pop("name"), "must have a name"),
            (lambda items: items[0].update(size=None), "integer size"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                assets, release = self._fixture(Path(directory))
                mutation(release["assets"])
                with self.assertRaisesRegex(ContractError, message):
                    self._result(assets, release)

    def test_result_hashes_exact_remote_asset_when_api_digest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            missing = release["assets"][0]
            missing["digest"] = None
            missing["url"] = "https://api.github.com/repos/example/repo/releases/assets/1"
            expected = hashlib.sha256((assets / missing["name"]).read_bytes()).hexdigest()
            with patch("release_contract._download_asset_digest", return_value=expected) as download:
                result = self._result(assets, release)
        download.assert_called_once_with(missing)
        emitted = {item["name"]: item["digest"] for item in result["release"]["assets"]}
        self.assertEqual(f"sha256:{expected}", emitted[missing["name"]])

    def test_result_rejects_arbitrary_remote_asset_url_when_digest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            missing = release["assets"][0]
            missing["digest"] = None
            missing["url"] = "https://attacker.example/payload"
            with self.assertRaisesRegex(
                ContractError, "exact api.github.com release-asset API URL"
            ):
                self._result(assets, release)

    def test_result_rejects_transaction_marker_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            release["body"] = release["body"].replace(
                "publication transaction: `", "publication transaction: `" + "b" * 64 + "`\n"
            )
            with self.assertRaisesRegex(
                ContractError, "transaction id does not match"
            ):
                self._result(assets, release)

    def test_result_rejects_publication_artifact_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            assets, release = self._fixture(Path(directory))
            release["body"] = release["body"].replace(
                "publication artifact digest: `sha256:" + "a" * 64 + "`",
                "publication artifact digest: `sha256:" + "b" * 64 + "`",
            )
            with self.assertRaisesRegex(
                ContractError, "missing exact evidence: publication artifact digest"
            ):
                self._result(assets, release)


if __name__ == "__main__":
    unittest.main()
