#!/usr/bin/env python3
"""Independently verify optional Windows CUDA pack archives without a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, BinaryIO, Mapping

from cuda_pack_contract import (
    CUDA_VARIANTS,
    CUOBJDUMP_SHA256,
    CUOBJDUMP_VERSION,
    CudaContractError,
    select_cuda_pack,
    validate_variant_metadata,
)


class VerificationError(RuntimeError):
    """Raised when a CUDA pack archive violates its contract."""


def stream_sha256(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise VerificationError(f"Unable to read pack member: {member.name}")
    with source:
        return source.read()


def verify_pack(
    path: Path,
    *,
    expected_tag: str,
    expected_commit: str,
) -> dict[str, Any]:
    """Verify one pack and return its validated manifest."""

    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise VerificationError(f"Pack contains duplicate members: {path.name}")
        for member in members:
            pure = PurePosixPath(member.name)
            if (
                not member.isfile()
                or pure.is_absolute()
                or len(pure.parts) != 1
                or pure.name in {"", ".", ".."}
            ):
                raise VerificationError(
                    f"Pack member is not a regular top-level file: {member.name}"
                )
        by_name = {member.name: member for member in members}
        metadata_member = by_name.get("cuda-pack.json")
        if metadata_member is None:
            raise VerificationError("Pack is missing cuda-pack.json")
        if metadata_member.size > 1024 * 1024:
            raise VerificationError("Pack manifest exceeds the 1 MiB safety limit")
        try:
            manifest = json.loads(_read_member(archive, metadata_member))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError("Pack manifest is not valid UTF-8 JSON") from error
        if not isinstance(manifest, dict):
            raise VerificationError("Pack manifest must be a JSON object")

        if manifest.get("contract_version") != 2:
            raise VerificationError("Pack contract version must be 2")
        if manifest.get("llama_cpp_tag") != expected_tag:
            raise VerificationError("Pack tag does not match the requested upstream tag")
        if manifest.get("llama_cpp_commit") != expected_commit:
            raise VerificationError("Pack commit does not match the requested upstream commit")
        if manifest.get("platform") != "windows" or manifest.get("arch") != "x64":
            raise VerificationError("Pack platform must be Windows x64")

        try:
            variant = validate_variant_metadata(manifest)
        except CudaContractError as error:
            raise VerificationError(str(error)) from error
        major = str(variant.cuda_major)
        expected_backend = f"ggml-cuda-{major}.dll"
        expected_payload = {
            expected_backend,
            f"cudart64_{major}.dll",
            f"cublas64_{major}.dll",
            f"cublasLt64_{major}.dll",
        }
        if manifest.get("backend_library") != expected_backend:
            raise VerificationError("Pack backend filename is invalid")
        if set(by_name) != expected_payload | {"cuda-pack.json"}:
            raise VerificationError("Pack contains missing or unexpected payload files")

        file_entries = manifest.get("files")
        if not isinstance(file_entries, list):
            raise VerificationError("Pack manifest files must be an array")
        entry_map: dict[str, Mapping[str, Any]] = {}
        for entry in file_entries:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                raise VerificationError("Pack manifest contains an invalid file entry")
            name = entry["name"]
            if name in entry_map:
                raise VerificationError(f"Duplicate manifest file entry: {name}")
            entry_map[name] = entry
        if set(entry_map) != expected_payload:
            raise VerificationError("Manifest payload list differs from archive payload")
        for name in sorted(expected_payload):
            member = by_name[name]
            entry = entry_map[name]
            if entry.get("size") != member.size:
                raise VerificationError(f"Manifest size mismatch for {name}")
            source = archive.extractfile(member)
            if source is None:
                raise VerificationError(f"Unable to hash pack member: {name}")
            with source:
                actual = stream_sha256(source)
            if entry.get("sha256") != actual:
                raise VerificationError(f"Manifest SHA-256 mismatch for {name}")

        inspector = manifest.get("device_code", {}).get("inspector", {})
        if (
            inspector.get("name") != "NVIDIA cuobjdump"
            or inspector.get("sha256") != CUOBJDUMP_SHA256
            or CUOBJDUMP_VERSION not in str(inspector.get("version", ""))
        ):
            raise VerificationError("Pack has incomplete fatbin inspector provenance")

        source_assets = manifest.get("source_assets", [])
        if not isinstance(source_assets, list) or len(source_assets) != 2:
            raise VerificationError("Pack must record exactly two source assets")
        expected_sources = {
            f"llama-{expected_tag}-bin-win-cuda-{variant.cuda_version}-x64.zip",
            f"cudart-llama-bin-win-cuda-{variant.cuda_version}-x64.zip",
        }
        source_names: set[str] = set()
        for entry in source_assets:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("name"), str):
                raise VerificationError("Pack contains an invalid source asset entry")
            name = entry["name"]
            source_names.add(name)
            if re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))) is None:
                raise VerificationError(f"Pack source asset has no SHA-256: {name}")
            expected_url = (
                "https://github.com/ggml-org/llama.cpp/releases/"
                f"download/{expected_tag}/{name}"
            )
            if entry.get("url") != expected_url:
                raise VerificationError(f"Pack source asset URL is invalid: {name}")
        if source_names != expected_sources:
            raise VerificationError("Pack source assets differ from exact upstream variant")
        return manifest


def verify_selection_policy(manifests: list[Mapping[str, Any]]) -> None:
    """Exercise important driver/architecture boundaries from pack metadata."""

    versions = [manifest.get("cuda_version") for manifest in manifests]
    if len(versions) != 2 or set(versions) != set(CUDA_VARIANTS):
        raise VerificationError("Selection verification requires CUDA 12.4 and 13.3")
    cases = (
        (49, 610, None),
        (50, 524, None),
        (50, 525, 12),
        (70, 610, 12),
        (75, 579, 12),
        (75, 580, 13),
        (120, 610, 13),
    )
    for compute_capability, driver_family, expected_major in cases:
        try:
            selected = select_cuda_pack(
                manifests,
                compute_capability=compute_capability,
                driver_family=driver_family,
            )
        except CudaContractError as error:
            raise VerificationError(str(error)) from error
        actual_major = None if selected is None else selected.get("cuda_major")
        if actual_major != expected_major:
            raise VerificationError(
                "CUDA selection policy mismatch for "
                f"CC {compute_capability}, driver {driver_family}: "
                f"expected {expected_major}, got {actual_major}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", action="append", required=True, type=Path)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifests = [
            verify_pack(
                path,
                expected_tag=args.expected_tag,
                expected_commit=args.expected_commit,
            )
            for path in args.pack
        ]
        verify_selection_policy(manifests)
    except (OSError, tarfile.TarError, VerificationError) as error:
        print(f"ERROR: {error}")
        return 1
    print(
        json.dumps(
            {
                "packs": [
                    {
                        "cuda_version": manifest["cuda_version"],
                        "compatibility": manifest["compatibility"],
                        "device_code": {
                            "ptx_architectures": manifest["device_code"][
                                "ptx_architectures"
                            ],
                            "sass_architectures": manifest["device_code"][
                                "sass_architectures"
                            ],
                        },
                    }
                    for manifest in manifests
                ],
                "selection_policy": "verified",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
