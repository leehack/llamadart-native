#!/usr/bin/env python3
"""Validate llamadart-native release tags and their upstream provenance."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


STABLE_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
STABLE_WRAPPER_RE = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"-llamadart\.([1-9][0-9]*)$"
)
NIGHTLY_RE = re.compile(r"^b(0|[1-9][0-9]*)$")
NIGHTLY_WRAPPER_RE = re.compile(
    r"^b(0|[1-9][0-9]*)-llamadart\.([1-9][0-9]*)$"
)


class PolicyError(ValueError):
    """Raised when a release tag violates the native version policy."""


@dataclass(frozen=True)
class Version:
    tag: str
    channel: str
    kind: str
    core: tuple[int, ...]
    rebuild: int = 0

    @property
    def github_prerelease(self) -> bool:
        return self.channel == "nightly" or self.kind == "wrapper"


def parse_upstream_ref(ref: str) -> Version:
    stable = STABLE_RE.fullmatch(ref)
    if stable:
        return Version(ref, "stable", "upstream", tuple(map(int, stable.groups())))

    nightly = NIGHTLY_RE.fullmatch(ref)
    if nightly:
        return Version(ref, "nightly", "upstream", (int(nightly.group(1)),))

    raise PolicyError(
        f"invalid llama.cpp release ref {ref!r}: expected stable vMAJOR.MINOR.PATCH "
        "or explicit nightly bNNNN"
    )


def parse_native_tag(tag: str) -> Version:
    stable = STABLE_RE.fullmatch(tag)
    if stable:
        return Version(tag, "stable", "upstream", tuple(map(int, stable.groups())))

    stable_wrapper = STABLE_WRAPPER_RE.fullmatch(tag)
    if stable_wrapper:
        major, minor, patch, rebuild = map(int, stable_wrapper.groups())
        return Version(tag, "stable", "wrapper", (major, minor, patch), rebuild)

    nightly = NIGHTLY_RE.fullmatch(tag)
    if nightly:
        return Version(tag, "nightly", "upstream", (int(nightly.group(1)),))

    nightly_wrapper = NIGHTLY_WRAPPER_RE.fullmatch(tag)
    if nightly_wrapper:
        build, rebuild = map(int, nightly_wrapper.groups())
        return Version(tag, "nightly", "wrapper", (build,), rebuild)

    raise PolicyError(
        f"invalid llamadart-native release tag {tag!r}: expected vMAJOR.MINOR.PATCH, "
        "vMAJOR.MINOR.NEXT_PATCH-llamadart.N, bNNNN, or bNNNN-llamadart.N"
    )


def wrapper_tag_for(upstream: Version, rebuild: int = 1) -> str:
    if rebuild < 1:
        raise PolicyError("wrapper rebuild number must be at least 1")
    if upstream.channel == "stable":
        major, minor, patch = upstream.core
        return f"v{major}.{minor}.{patch + 1}-llamadart.{rebuild}"
    return f"b{upstream.core[0]}-llamadart.{rebuild}"


def validate_pair(upstream_ref: str, native_tag: str) -> tuple[Version, Version]:
    upstream = parse_upstream_ref(upstream_ref)
    native = parse_native_tag(native_tag)

    if native.channel != upstream.channel:
        raise PolicyError(
            f"native tag {native_tag!r} is a {native.channel} tag but upstream "
            f"ref {upstream_ref!r} is {upstream.channel}"
        )

    if native.kind == "upstream":
        if native.tag != upstream.tag:
            raise PolicyError(
                f"upstream-aligned native tag must exactly match {upstream.tag!r}; "
                f"got {native.tag!r}"
            )
        return upstream, native

    if upstream.channel == "stable":
        major, minor, patch = upstream.core
        if native.core != (major, minor, patch + 1):
            raise PolicyError(
                f"wrapper-only rebuild for {upstream.tag} must use "
                f"{wrapper_tag_for(upstream)} (then increment N); got {native.tag!r}"
            )
    elif native.core != upstream.core:
        raise PolicyError(
            f"wrapper-only rebuild for {upstream.tag} must use "
            f"{wrapper_tag_for(upstream)} (then increment N); got {native.tag!r}"
        )

    return upstream, native


def validate_automatic_upstream(upstream: Version) -> None:
    if upstream.channel != "stable":
        raise PolicyError(
            "automatic release discovery requires stable vMAJOR.MINOR.PATCH; "
            f"got {upstream.tag!r}"
        )


def _stable_precedence(version: Version) -> tuple[int, int, int, int, int]:
    if version.channel != "stable":
        raise PolicyError(f"{version.tag!r} is not a stable-channel native tag")
    major, minor, patch = version.core
    # A wrapper rebuild is a SemVer prerelease of the next patch. It sorts after
    # the upstream version it rebuilds and before the eventual normal release.
    final = 1 if version.kind == "upstream" else 0
    return major, minor, patch, final, version.rebuild


def validate_history(candidate: Version, existing_tags: Iterable[str]) -> None:
    existing: list[Version] = []
    for tag in existing_tags:
        tag = tag.strip()
        if not tag:
            continue
        if tag == candidate.tag:
            raise PolicyError(
                f"release tag collision: {candidate.tag!r} already exists; use a new "
                "wrapper rebuild tag instead of mutating an immutable release"
            )
        try:
            existing.append(parse_native_tag(tag))
        except PolicyError:
            # Historical tags outside this contract remain immutable but do not
            # participate in ordering for the supported channels.
            continue

    if candidate.channel == "stable":
        stable = [version for version in existing if version.channel == "stable"]
        if stable:
            latest = max(stable, key=_stable_precedence)
            if _stable_precedence(candidate) < _stable_precedence(latest):
                raise PolicyError(
                    f"stable-channel rollback: {candidate.tag!r} precedes existing "
                    f"native release {latest.tag!r}"
                )
        return

    same_nightly = [
        version
        for version in existing
        if version.channel == "nightly" and version.core == candidate.core
    ]
    if same_nightly:
        latest_rebuild = max(version.rebuild for version in same_nightly)
        if candidate.rebuild < latest_rebuild:
            raise PolicyError(
                f"nightly wrapper rollback: {candidate.tag!r} precedes an existing "
                f"rebuild for b{candidate.core[0]}"
            )


def manifest_native_tag(manifest: Mapping[str, Any]) -> str:
    """Return the native tag from current or immutable legacy manifests."""
    legacy_tag = manifest.get("tag")
    native_tag = manifest.get("native_release_tag", legacy_tag)
    if not isinstance(native_tag, str) or not native_tag:
        raise PolicyError("manifest is missing native_release_tag/tag")
    if legacy_tag is not None and legacy_tag != native_tag:
        raise PolicyError("manifest native_release_tag does not match legacy tag alias")
    return native_tag


def _load_existing(path: Path | None) -> list[str]:
    return [] if path is None else path.read_text().splitlines()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-ref", required=True)
    parser.add_argument("--native-tag", required=True)
    parser.add_argument("--existing-tags-file", type=Path)
    parser.add_argument("--require-stable-upstream", action="store_true")
    args = parser.parse_args()

    try:
        upstream, native = validate_pair(args.upstream_ref, args.native_tag)
        if args.require_stable_upstream:
            validate_automatic_upstream(upstream)
        validate_history(native, _load_existing(args.existing_tags_file))
    except PolicyError as error:
        parser.error(str(error))

    print(f"upstream_channel={upstream.channel}")
    print(f"release_kind={native.kind}")
    print(f"github_prerelease={'true' if native.github_prerelease else 'false'}")
    print(f"wrapper_example={wrapper_tag_for(upstream)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
