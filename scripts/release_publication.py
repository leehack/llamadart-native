#!/usr/bin/env python3
"""Publish an immutable native release as a retry-safe draft-first transaction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping
from urllib.parse import urlparse


class PublicationError(RuntimeError):
    """Raised when existing GitHub state does not match the approved release."""


ARTIFACT_DIGEST_RE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})$")
WORKFLOW_RUN_PATH_RE = re.compile(
    r"^/[^/]+/[^/]+/actions/runs/[1-9][0-9]*(?:/attempts/[1-9][0-9]*)?$"
)


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str
    path: Path | None = None


@dataclass(frozen=True)
class DesiredRelease:
    tag: str
    native_commit: str
    prerelease: bool
    name: str
    body: str
    assets: Mapping[str, Asset]
    transaction_id: str


@dataclass(frozen=True)
class ExistingRelease:
    draft: bool
    prerelease: bool
    name: str
    body: str
    assets: Mapping[str, Asset]


@dataclass(frozen=True)
class PublicationPlan:
    create_tag: bool
    create_draft_release: bool
    upload_assets: tuple[str, ...]
    publish_draft: bool
    complete: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_desired_release(
    *,
    tag: str,
    native_commit: str,
    upstream_ref: str,
    upstream_commit: str,
    prerelease: bool,
    assets_dir: Path,
    workflow_run_url: str,
    artifact_digest: str,
) -> DesiredRelease:
    digest_match = ARTIFACT_DIGEST_RE.fullmatch(artifact_digest)
    if digest_match is None:
        raise PublicationError(
            "publication artifact digest must be a SHA-256 value"
        )
    normalized_artifact_digest = f"sha256:{digest_match.group(1).lower()}"

    parsed_run_url = urlparse(workflow_run_url)
    if (
        parsed_run_url.scheme != "https"
        or not parsed_run_url.netloc
        or parsed_run_url.params
        or parsed_run_url.query
        or parsed_run_url.fragment
        or WORKFLOW_RUN_PATH_RE.fullmatch(parsed_run_url.path) is None
    ):
        raise PublicationError(
            "workflow run URL must identify an exact HTTPS GitHub Actions run"
        )

    assets: dict[str, Asset] = {}
    for path in sorted(assets_dir.iterdir()):
        if not path.is_file():
            continue
        assets[path.name] = Asset(path.name, path.stat().st_size, _sha256(path), path)
    if not assets:
        raise PublicationError(f"no release assets found in {assets_dir}")

    transaction_payload = {
        "native_release_tag": tag,
        "native_commit": native_commit,
        "llama_cpp_tag": upstream_ref,
        "llama_cpp_commit": upstream_commit,
        "prerelease": prerelease,
        "workflow_run_url": workflow_run_url,
        "artifact_digest": normalized_artifact_digest,
        "assets": {
            name: {"sha256": asset.sha256, "size": asset.size}
            for name, asset in assets.items()
        },
    }
    transaction_id = hashlib.sha256(
        json.dumps(
            transaction_payload, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    body = (
        f"llamadart-native tag: `{tag}`\n"
        f"llama.cpp tag/ref: `{upstream_ref}`\n"
        f"llama.cpp commit: `{upstream_commit}`\n"
        f"llamadart-native commit: `{native_commit}`\n"
        f"publication transaction: `{transaction_id}`\n"
        f"workflow run: {workflow_run_url}\n"
        f"publication artifact digest: `{normalized_artifact_digest}`\n"
    )
    return DesiredRelease(
        tag=tag,
        native_commit=native_commit,
        prerelease=prerelease,
        name=tag,
        body=body,
        assets=assets,
        transaction_id=transaction_id,
    )


def reconcile_publication(
    desired: DesiredRelease,
    *,
    tag_target: str | None,
    release: ExistingRelease | None,
) -> PublicationPlan:
    if tag_target is not None and tag_target != desired.native_commit:
        raise PublicationError(
            f"immutable tag {desired.tag!r} points to {tag_target}, expected "
            f"{desired.native_commit}"
        )
    if release is not None and tag_target is None:
        raise PublicationError(
            f"release {desired.tag!r} exists without its approved immutable tag"
        )

    existing_assets: Mapping[str, Asset] = {}
    if release is not None:
        if release.name != desired.name:
            raise PublicationError(
                f"release name mismatch for {desired.tag!r}: {release.name!r}"
            )
        if release.body.rstrip("\r\n") != desired.body.rstrip("\r\n"):
            raise PublicationError(
                f"release body/correlation mismatch for {desired.tag!r}"
            )
        if release.prerelease != desired.prerelease:
            raise PublicationError(
                f"release prerelease mismatch for {desired.tag!r}: expected "
                f"{desired.prerelease}"
            )
        existing_assets = release.assets

        unexpected = sorted(set(existing_assets) - set(desired.assets))
        if unexpected:
            raise PublicationError(
                f"release {desired.tag!r} has unexpected assets: {unexpected}"
            )
        for name, existing in existing_assets.items():
            expected = desired.assets[name]
            if existing.size != expected.size or existing.sha256 != expected.sha256:
                raise PublicationError(
                    f"immutable asset mismatch for {desired.tag!r}/{name}: "
                    f"expected {expected.sha256}/{expected.size}, got "
                    f"{existing.sha256}/{existing.size}"
                )

        if not release.draft and set(existing_assets) != set(desired.assets):
            missing = sorted(set(desired.assets) - set(existing_assets))
            raise PublicationError(
                f"published release {desired.tag!r} is incomplete; missing {missing}"
            )

    missing_assets = tuple(sorted(set(desired.assets) - set(existing_assets)))
    complete = (
        release is not None
        and not release.draft
        and not missing_assets
        and tag_target == desired.native_commit
    )
    return PublicationPlan(
        create_tag=tag_target is None,
        create_draft_release=release is None,
        upload_assets=missing_assets,
        publish_draft=release is None or (release.draft and not complete),
        complete=complete,
    )


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=capture, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "command failed").strip()
        raise PublicationError(f"{' '.join(command)}: {detail}")
    return result


def _remote_tag_target(repository: str, tag: str) -> str | None:
    result = _run(
        [
            "git",
            "ls-remote",
            f"https://github.com/{repository}.git",
            f"refs/tags/{tag}",
            f"refs/tags/{tag}^{{}}",
        ],
        capture=True,
    )
    refs = {}
    for line in result.stdout.splitlines():
        sha, ref = line.split(maxsplit=1)
        refs[ref] = sha
    return refs.get(f"refs/tags/{tag}^{{}}", refs.get(f"refs/tags/{tag}"))


def _release_json(repository: str, tag: str) -> dict[str, Any] | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repository}/releases/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    detail = (result.stderr or result.stdout).strip()
    if "HTTP 404" in detail or "Not Found" in detail:
        return None
    raise PublicationError(f"unable to read release {tag!r}: {detail}")


def _download_asset_digest(asset: Mapping[str, Any]) -> str:
    with tempfile.NamedTemporaryFile() as output:
        result = subprocess.run(
            [
                "gh",
                "api",
                str(asset["url"]),
                "-H",
                "Accept: application/octet-stream",
            ],
            stdout=output,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise PublicationError(
                f"unable to download existing asset {asset['name']!r}: "
                f"{result.stderr.decode(errors='replace').strip()}"
            )
        output.flush()
        return _sha256(Path(output.name))


def _existing_release(payload: Mapping[str, Any] | None) -> ExistingRelease | None:
    if payload is None:
        return None
    assets: dict[str, Asset] = {}
    for item in payload.get("assets", []):
        digest = item.get("digest")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            sha256 = digest.removeprefix("sha256:")
        else:
            sha256 = _download_asset_digest(item)
        assets[item["name"]] = Asset(item["name"], int(item["size"]), sha256)
    return ExistingRelease(
        draft=bool(payload["draft"]),
        prerelease=bool(payload["prerelease"]),
        name=str(payload.get("name") or ""),
        body=str(payload.get("body") or ""),
        assets=assets,
    )


def _ensure_tag(repository: str, desired: DesiredRelease) -> None:
    target = _remote_tag_target(repository, desired.tag)
    if target is not None:
        if target != desired.native_commit:
            raise PublicationError(
                f"immutable tag {desired.tag!r} points to {target}, expected "
                f"{desired.native_commit}"
            )
        return

    local = subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{desired.tag}^{{commit}}"],
        capture_output=True,
        text=True,
    )
    if local.returncode == 0:
        if local.stdout.strip() != desired.native_commit:
            raise PublicationError(
                f"local tag {desired.tag!r} conflicts with approved native commit"
            )
    else:
        _run(["git", "tag", desired.tag, desired.native_commit])

    pushed = subprocess.run(
        ["git", "push", "origin", f"refs/tags/{desired.tag}"],
        capture_output=True,
        text=True,
    )
    if pushed.returncode != 0:
        raced_target = _remote_tag_target(repository, desired.tag)
        if raced_target != desired.native_commit:
            detail = (pushed.stderr or pushed.stdout).strip()
            raise PublicationError(f"unable to create immutable tag: {detail}")


def publish(repository: str, desired: DesiredRelease) -> None:
    tag_target = _remote_tag_target(repository, desired.tag)
    payload = _release_json(repository, desired.tag)
    plan = reconcile_publication(
        desired, tag_target=tag_target, release=_existing_release(payload)
    )
    if plan.complete:
        print(f"Release {desired.tag} already matches transaction {desired.transaction_id}.")
        return

    if plan.create_tag:
        _ensure_tag(repository, desired)

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as notes:
        notes.write(desired.body)
        notes.flush()

        if plan.create_draft_release:
            command = [
                "gh",
                "release",
                "create",
                desired.tag,
                "--repo",
                repository,
                "--verify-tag",
                "--draft",
                "--title",
                desired.name,
                "--notes-file",
                notes.name,
            ]
            if desired.prerelease:
                command.append("--prerelease")
            created = subprocess.run(command, capture_output=True, text=True)
            if created.returncode != 0:
                # A concurrent retry may have created the same draft. The exact
                # state is revalidated below before any asset is accepted.
                if _release_json(repository, desired.tag) is None:
                    detail = (created.stderr or created.stdout).strip()
                    raise PublicationError(f"unable to create draft release: {detail}")

        current = _existing_release(_release_json(repository, desired.tag))
        current_plan = reconcile_publication(
            desired,
            tag_target=_remote_tag_target(repository, desired.tag),
            release=current,
        )
        for name in current_plan.upload_assets:
            asset = desired.assets[name]
            if asset.path is None:
                raise PublicationError(f"missing local path for desired asset {name}")
            _run(
                [
                    "gh",
                    "release",
                    "upload",
                    desired.tag,
                    str(asset.path),
                    "--repo",
                    repository,
                ]
            )

        verified = _existing_release(_release_json(repository, desired.tag))
        verified_plan = reconcile_publication(
            desired,
            tag_target=_remote_tag_target(repository, desired.tag),
            release=verified,
        )
        if verified_plan.upload_assets:
            raise PublicationError(
                f"draft release is still missing assets: {verified_plan.upload_assets}"
            )
        if verified is not None and verified.draft:
            _run(
                [
                    "gh",
                    "release",
                    "edit",
                    desired.tag,
                    "--repo",
                    repository,
                    "--draft=false",
                    f"--prerelease={'true' if desired.prerelease else 'false'}",
                    "--title",
                    desired.name,
                    "--notes-file",
                    notes.name,
                ]
            )

    final = reconcile_publication(
        desired,
        tag_target=_remote_tag_target(repository, desired.tag),
        release=_existing_release(_release_json(repository, desired.tag)),
    )
    if not final.complete:
        raise PublicationError(f"release {desired.tag!r} did not reach exact complete state")
    print(f"Published exact immutable transaction {desired.transaction_id}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--native-commit", required=True)
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--prerelease", required=True, choices=("true", "false"))
    parser.add_argument("--workflow-run-url", required=True)
    parser.add_argument("--artifact-digest", required=True)
    args = parser.parse_args()

    try:
        desired = build_desired_release(
            tag=args.tag,
            native_commit=args.native_commit,
            upstream_ref=args.upstream_ref,
            upstream_commit=args.upstream_commit,
            prerelease=args.prerelease == "true",
            assets_dir=args.assets_dir,
            workflow_run_url=args.workflow_run_url,
            artifact_digest=args.artifact_digest,
        )
        publish(args.repository, desired)
    except PublicationError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
