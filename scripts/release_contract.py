#!/usr/bin/env python3
"""Validate and emit machine-readable native release orchestration contracts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.parse import urlparse

from release_version_policy import PolicyError, validate_pair


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
CORRELATION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
ARTIFACT_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$", re.IGNORECASE)
GITHUB_ASSET_PATH_RE = re.compile(
    r"^/repos/[^/\s]+/[^/\s]+/releases/assets/[1-9][0-9]*$"
)
DISCOVERY_STATUSES = ("candidate", "noop", "incompatible")
SMOKE_POLICIES = ("required", "skip")
SMOKE_CONCLUSIONS = ("passed", "skipped")
RUNTIME_BUNDLES = (
    "android-arm64",
    "android-x64",
    "ios-arm64",
    "ios-arm64-sim",
    "ios-x86_64-sim",
    "linux-arm64",
    "linux-x64",
    "macos-arm64",
    "macos-x86_64",
    "windows-arm64",
    "windows-x64",
)


class ContractError(ValueError):
    """Raised when an orchestration input or result is ambiguous."""


def _full_commit(value: str, label: str) -> str:
    if COMMIT_RE.fullmatch(value) is None:
        raise ContractError(f"{label} must be a full 40-hex commit SHA")
    return value.lower()


def _correlation(value: str) -> str:
    if CORRELATION_RE.fullmatch(value) is None:
        raise ContractError(
            "correlation identifier must be 1-128 safe characters, start with an "
            "alphanumeric character, and contain only alphanumerics, '.', '_', ':', '/', or '-'"
        )
    return value


def _artifact_digest(value: str) -> str:
    match = ARTIFACT_DIGEST_RE.fullmatch(value)
    if match is None:
        raise ContractError("publication artifact digest must be a SHA-256 value")
    return f"sha256:{match.group(1).lower()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_asset_digest(asset: Mapping[str, Any]) -> str:
    url = asset.get("url")
    if not isinstance(url, str) or not url:
        raise ContractError(
            f"GitHub asset {asset.get('name')!r} has neither a digest nor an API URL"
        )
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "api.github.com"
        or parsed_url.params
        or parsed_url.query
        or parsed_url.fragment
        or GITHUB_ASSET_PATH_RE.fullmatch(parsed_url.path) is None
    ):
        raise ContractError(
            f"GitHub asset {asset.get('name')!r} must use an exact GitHub "
            "release-asset API URL"
        )
    with tempfile.NamedTemporaryFile() as output:
        result = subprocess.run(
            ["gh", "api", url, "-H", "Accept: application/octet-stream"],
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ContractError(
                f"unable to download GitHub asset {asset.get('name')!r}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        output.flush()
        return _sha256(Path(output.name))


def validate_dispatch(
    *,
    upstream_ref: str,
    upstream_commit: str,
    native_release_tag: str,
    smoke_policy: str,
    correlation_id: str,
    publish_release: bool,
) -> dict[str, str]:
    """Validate the immutable fields required from an approved orchestrator."""
    try:
        upstream, native = validate_pair(upstream_ref, native_release_tag)
    except PolicyError as error:
        raise ContractError(str(error)) from error
    commit = _full_commit(upstream_commit, "upstream commit")
    if smoke_policy not in SMOKE_POLICIES:
        raise ContractError(f"smoke policy must be one of {SMOKE_POLICIES}")
    if publish_release and smoke_policy != "required":
        raise ContractError("publication requires smoke_policy=required")
    return {
        "llama_cpp_ref": upstream.tag,
        "llama_cpp_commit": commit,
        "native_release_tag": native.tag,
        "tag": native.tag,
        "upstream_channel": upstream.channel,
        "release_kind": native.kind,
        "github_prerelease": "true" if native.github_prerelease else "false",
        "smoke_policy": smoke_policy,
        "correlation_id": _correlation(correlation_id),
    }


def build_discovery_report(
    *,
    status: str,
    upstream_ref: str,
    upstream_commit: str,
    native_head_commit: str,
    workflow_run_id: str,
    workflow_run_url: str,
    message: str,
) -> dict[str, Any]:
    if status not in DISCOVERY_STATUSES:
        raise ContractError(f"discovery status must be one of {DISCOVERY_STATUSES}")
    if not workflow_run_id.isdigit() or workflow_run_id.startswith("0"):
        raise ContractError("workflow run id must be a positive integer")
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "candidate": status == "candidate",
        "native_release_tag": upstream_ref,
        "tag": upstream_ref,
        "llama_cpp_ref": upstream_ref,
        "llama_cpp_commit": _full_commit(upstream_commit, "upstream commit"),
        "native_head_commit": _full_commit(native_head_commit, "native head commit"),
        "workflow_run": {"id": int(workflow_run_id), "url": workflow_run_url},
        "message": message,
    }
    if status != "incompatible":
        try:
            validate_pair(upstream_ref, upstream_ref)
        except PolicyError as error:
            raise ContractError(str(error)) from error
    return report


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ContractError(f"invalid SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        if name in checksums:
            raise ContractError(f"duplicate checksum entry: {name}")
        checksums[name] = digest
    return checksums


def _expected_assets(tag: str) -> set[str]:
    assets = {
        f"llamadart-native-{bundle}-{tag}.tar.gz" for bundle in RUNTIME_BUNDLES
    }
    assets.update(
        {
            f"llamadart-native-apple-xcframework-{tag}.zip",
            f"llamadart-native-headers-{tag}.tar.gz",
        }
    )
    return assets


def _manifest_asset_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError("manifest artifacts must be a list")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractError(f"manifest artifact {index} must be a JSON object")
        name = item.get("file")
        digest = item.get("sha256")
        size = item.get("size")
        if not isinstance(name, str) or not name:
            raise ContractError(f"manifest artifact {index} must have a file name")
        if name in names:
            raise ContractError(f"duplicate manifest artifact: {name}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ContractError(f"manifest artifact {name} must have a SHA-256 digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(f"manifest artifact {name} must have a non-negative integer size")
        names.add(name)
        entries.append(dict(item))
    return entries


def _remote_asset_entries(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ContractError("GitHub release metadata must include an asset list")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ContractError(f"GitHub release asset {index} must be a JSON object")
        name = item.get("name")
        size = item.get("size")
        if not isinstance(name, str) or not name:
            raise ContractError(f"GitHub release asset {index} must have a name")
        if name in names:
            raise ContractError(f"duplicate GitHub release asset: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ContractError(
                f"GitHub release asset {name} must have a non-negative integer size"
            )
        names.add(name)
        entries.append(dict(item))
    return entries


def build_release_result(
    *,
    assets_dir: Path,
    release_metadata: Mapping[str, Any] | None,
    native_release_tag: str,
    upstream_ref: str,
    upstream_commit: str,
    native_commit: str,
    correlation_id: str,
    smoke_policy: str,
    smoke_conclusion: str,
    publish_release: bool,
    workflow_repository: str,
    workflow_run_id: str,
    workflow_run_attempt: str,
    workflow_run_url: str,
    workflow_head_sha: str,
    publication_artifact_id: str,
    publication_artifact_url: str,
    publication_artifact_digest: str,
) -> dict[str, Any]:
    contract = validate_dispatch(
        upstream_ref=upstream_ref,
        upstream_commit=upstream_commit,
        native_release_tag=native_release_tag,
        smoke_policy=smoke_policy,
        correlation_id=correlation_id,
        publish_release=publish_release,
    )
    native_commit = _full_commit(native_commit, "native commit")
    workflow_head_sha = _full_commit(workflow_head_sha, "workflow head SHA")
    if (
        not workflow_run_id.isdigit()
        or workflow_run_id.startswith("0")
        or not workflow_run_attempt.isdigit()
        or workflow_run_attempt.startswith("0")
        or not publication_artifact_id.isdigit()
        or publication_artifact_id.startswith("0")
    ):
        raise ContractError(
            "workflow run id/attempt and publication artifact id must be positive integers"
        )
    if smoke_conclusion not in SMOKE_CONCLUSIONS:
        raise ContractError(f"smoke conclusion must be one of {SMOKE_CONCLUSIONS}")
    if smoke_policy == "required" and smoke_conclusion != "passed":
        raise ContractError("required smoke policy must conclude passed")
    if smoke_policy == "skip" and smoke_conclusion != "skipped":
        raise ContractError("skip smoke policy must conclude skipped")

    manifest_path = assets_dir / "assets.json"
    checksums_path = assets_dir / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise ContractError("release assets must contain assets.json and SHA256SUMS")
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ContractError(f"assets.json is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise ContractError("assets.json must contain a JSON object")
    if manifest.get("native_release_tag") != native_release_tag or manifest.get("tag") != native_release_tag:
        raise ContractError("manifest native_release_tag/tag does not match dispatch")
    for field, expected in {
        "llama_cpp_tag": upstream_ref,
        "llama_cpp_commit": contract["llama_cpp_commit"],
        "native_commit": native_commit,
        "correlation_id": correlation_id,
        "smoke_policy": smoke_policy,
        "smoke_conclusion": smoke_conclusion,
    }.items():
        if manifest.get(field) != expected:
            raise ContractError(f"manifest {field} does not match dispatch/result")

    expected_assets = _expected_assets(native_release_tag)
    manifest_assets = _manifest_asset_entries(manifest.get("artifacts"))
    manifest_names = {item["file"] for item in manifest_assets}
    if manifest_names != expected_assets:
        raise ContractError(
            f"release bundle coverage mismatch: expected {sorted(expected_assets)}, got {sorted(manifest_names)}"
        )

    checksums = _read_checksums(checksums_path)
    if set(checksums) != expected_assets:
        raise ContractError("SHA256SUMS asset set does not match required bundle coverage")
    for item in manifest_assets:
        name = item["file"]
        asset_path = assets_dir / name
        if not asset_path.is_file():
            raise ContractError(f"missing release asset: {name}")
        digest = _sha256(asset_path)
        if checksums[name] != digest or item.get("sha256") != digest:
            raise ContractError(f"digest mismatch for release asset: {name}")
        if item.get("size") != asset_path.stat().st_size:
            raise ContractError(f"size mismatch for release asset: {name}")

    release: dict[str, Any] | None = None
    if publish_release:
        if not isinstance(release_metadata, Mapping):
            raise ContractError("published result requires exact GitHub release metadata")
        if release_metadata.get("tag_name") != native_release_tag:
            raise ContractError("GitHub release tag does not match dispatch")
        if release_metadata.get("draft") is not False:
            raise ContractError("GitHub release must be published, not draft")
        if bool(release_metadata.get("prerelease")) != (
            contract["github_prerelease"] == "true"
        ):
            raise ContractError("GitHub release prerelease classification does not match policy")
        body = release_metadata.get("body")
        if not isinstance(body, str):
            raise ContractError("GitHub release metadata must include its provenance body")
        transaction = re.search(r"publication transaction: `([0-9a-f]{64})`", body)
        for evidence in (
            f"orchestrator correlation: `{correlation_id}`",
            f"workflow run: {workflow_run_url}",
            f"workflow head SHA: `{workflow_head_sha}`",
        ):
            if evidence not in body:
                raise ContractError(f"GitHub release body is missing exact evidence: {evidence}")
        if transaction is None:
            raise ContractError("GitHub release body is missing publication transaction id")
        remote_assets = _remote_asset_entries(release_metadata.get("assets"))
        remote_names = {item["name"] for item in remote_assets}
        expected_remote = expected_assets | {"assets.json", "SHA256SUMS"}
        if remote_names != expected_remote:
            raise ContractError("GitHub release asset set does not match verified publication")
        local_digests = checksums | {
            "assets.json": _sha256(manifest_path),
            "SHA256SUMS": _sha256(checksums_path),
        }
        verified_remote_digests: dict[str, str] = {}
        for item in remote_assets:
            name = item["name"]
            digest = item.get("digest")
            if isinstance(digest, str) and ARTIFACT_DIGEST_RE.fullmatch(digest):
                verified_digest = _artifact_digest(digest).removeprefix("sha256:")
            else:
                verified_digest = _download_asset_digest(item)
            if verified_digest != local_digests[name]:
                raise ContractError(f"GitHub asset digest mismatch for {name}")
            verified_remote_digests[name] = f"sha256:{verified_digest}"
            if item["size"] != (assets_dir / name).stat().st_size:
                raise ContractError(f"GitHub asset size mismatch for {name}")
        release = {
            "id": release_metadata.get("id"),
            "url": release_metadata.get("html_url"),
            "tag": release_metadata.get("tag_name"),
            "draft": release_metadata.get("draft"),
            "prerelease": release_metadata.get("prerelease"),
            "published_at": release_metadata.get("published_at"),
            "transaction_id": transaction.group(1),
            "assets": [
                {
                    "id": item.get("id"),
                    "name": item.get("name"),
                    "url": item.get("browser_download_url"),
                    "size": item.get("size"),
                    "digest": verified_remote_digests[item["name"]],
                }
                for item in sorted(remote_assets, key=lambda value: value["name"])
            ],
        }

    result = {
        "schema_version": 1,
        "status": "published" if publish_release else "prepared",
        "correlation_id": correlation_id,
        "native_release_tag": native_release_tag,
        "tag": native_release_tag,
        "llama_cpp_ref": upstream_ref,
        "llama_cpp_commit": contract["llama_cpp_commit"],
        "native_commit": native_commit,
        "workflow_run": {
            "repository": workflow_repository,
            "id": int(workflow_run_id),
            "attempt": int(workflow_run_attempt),
            "url": workflow_run_url,
            "head_sha": workflow_head_sha,
        },
        "publication_artifact": {
            "id": int(publication_artifact_id),
            "url": publication_artifact_url,
            "digest": _artifact_digest(publication_artifact_digest),
        },
        "manifest": {"sha256": _sha256(manifest_path)},
        "checksums": {"sha256": _sha256(checksums_path), "entries": checksums},
        "bundle_coverage": {
            "expected": list(RUNTIME_BUNDLES),
            "actual": list(RUNTIME_BUNDLES),
            "complete": True,
        },
        "smoke": {"policy": smoke_policy, "conclusion": smoke_conclusion},
        "release": release,
    }
    return result


def _write_json(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dispatch = subparsers.add_parser("validate-dispatch")
    dispatch.add_argument("--upstream-ref", required=True)
    dispatch.add_argument("--upstream-commit", required=True)
    dispatch.add_argument("--native-release-tag", required=True)
    dispatch.add_argument("--smoke-policy", required=True, choices=SMOKE_POLICIES)
    dispatch.add_argument("--correlation-id", required=True)
    dispatch.add_argument("--publish-release", required=True, choices=("true", "false"))

    discovery = subparsers.add_parser("discovery-report")
    discovery.add_argument("--status", required=True, choices=DISCOVERY_STATUSES)
    discovery.add_argument("--upstream-ref", required=True)
    discovery.add_argument("--upstream-commit", required=True)
    discovery.add_argument("--native-head-commit", required=True)
    discovery.add_argument("--workflow-run-id", required=True)
    discovery.add_argument("--workflow-run-url", required=True)
    discovery.add_argument("--message", required=True)
    discovery.add_argument("--output", required=True, type=Path)

    result = subparsers.add_parser("release-result")
    result.add_argument("--assets-dir", required=True, type=Path)
    result.add_argument("--release-metadata", type=Path)
    result.add_argument("--native-release-tag", required=True)
    result.add_argument("--upstream-ref", required=True)
    result.add_argument("--upstream-commit", required=True)
    result.add_argument("--native-commit", required=True)
    result.add_argument("--correlation-id", required=True)
    result.add_argument("--smoke-policy", required=True, choices=SMOKE_POLICIES)
    result.add_argument("--smoke-conclusion", required=True, choices=SMOKE_CONCLUSIONS)
    result.add_argument("--publish-release", required=True, choices=("true", "false"))
    result.add_argument("--workflow-repository", required=True)
    result.add_argument("--workflow-run-id", required=True)
    result.add_argument("--workflow-run-attempt", required=True)
    result.add_argument("--workflow-run-url", required=True)
    result.add_argument("--workflow-head-sha", required=True)
    result.add_argument("--publication-artifact-id", required=True)
    result.add_argument("--publication-artifact-url", required=True)
    result.add_argument("--publication-artifact-digest", required=True)
    result.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    try:
        if args.command == "validate-dispatch":
            payload = validate_dispatch(
                upstream_ref=args.upstream_ref,
                upstream_commit=args.upstream_commit,
                native_release_tag=args.native_release_tag,
                smoke_policy=args.smoke_policy,
                correlation_id=args.correlation_id,
                publish_release=args.publish_release == "true",
            )
            for key, value in payload.items():
                print(f"{key}={value}")
        elif args.command == "discovery-report":
            _write_json(
                build_discovery_report(
                    status=args.status,
                    upstream_ref=args.upstream_ref,
                    upstream_commit=args.upstream_commit,
                    native_head_commit=args.native_head_commit,
                    workflow_run_id=args.workflow_run_id,
                    workflow_run_url=args.workflow_run_url,
                    message=args.message,
                ),
                args.output,
            )
        else:
            metadata = None
            if args.release_metadata is not None:
                metadata = json.loads(args.release_metadata.read_text())
            _write_json(
                build_release_result(
                    assets_dir=args.assets_dir,
                    release_metadata=metadata,
                    native_release_tag=args.native_release_tag,
                    upstream_ref=args.upstream_ref,
                    upstream_commit=args.upstream_commit,
                    native_commit=args.native_commit,
                    correlation_id=args.correlation_id,
                    smoke_policy=args.smoke_policy,
                    smoke_conclusion=args.smoke_conclusion,
                    publish_release=args.publish_release == "true",
                    workflow_repository=args.workflow_repository,
                    workflow_run_id=args.workflow_run_id,
                    workflow_run_attempt=args.workflow_run_attempt,
                    workflow_run_url=args.workflow_run_url,
                    workflow_head_sha=args.workflow_head_sha,
                    publication_artifact_id=args.publication_artifact_id,
                    publication_artifact_url=args.publication_artifact_url,
                    publication_artifact_digest=args.publication_artifact_digest,
                ),
                args.output,
            )
    except ContractError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
