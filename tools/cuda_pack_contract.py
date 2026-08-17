#!/usr/bin/env python3
"""Shared, GPU-less compatibility contracts for optional Windows CUDA packs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping


class CudaContractError(RuntimeError):
    """Raised when CUDA device code or compatibility metadata is invalid."""


CUOBJDUMP_VERSION = "13.3.29"
CUOBJDUMP_SHA256 = "b6f56c1eb5edd046949f9c947e730a1bf0ed5beff6fc20f8ccafd8a1f5d2eff1"


@dataclass(frozen=True)
class CudaVariant:
    cuda_version: str
    cuda_major: int
    minimum_compute_capability: int
    minimum_driver_family: int
    minimum_driver_api: int
    ptx_architectures: frozenset[str]
    sass_architectures: frozenset[str]


# These are the exact GGML_NATIVE=OFF defaults in llama.cpp b10453 for the two
# upstream Windows release variants. Treat changes as an intentional contract
# update instead of silently widening or narrowing supported hardware.
CUDA_VARIANTS: dict[str, CudaVariant] = {
    "12.4": CudaVariant(
        cuda_version="12.4",
        cuda_major=12,
        minimum_compute_capability=50,
        minimum_driver_family=525,
        minimum_driver_api=12000,
        ptx_architectures=frozenset({"50", "61", "70", "75", "80", "90"}),
        sass_architectures=frozenset({"86", "89"}),
    ),
    "13.3": CudaVariant(
        cuda_version="13.3",
        cuda_major=13,
        minimum_compute_capability=75,
        minimum_driver_family=580,
        minimum_driver_api=13000,
        ptx_architectures=frozenset({"75", "80", "90"}),
        sass_architectures=frozenset({"86", "89", "120a", "121a"}),
    ),
}


_ARCHITECTURE_PATTERN = re.compile(r"\bsm_([0-9]+[a-z]?)\b", re.IGNORECASE)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_cuobjdump(cuobjdump: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            [str(cuobjdump), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CudaContractError(f"Unable to run cuobjdump: {error}") from error
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        detail = output.strip() or f"exit code {result.returncode}"
        raise CudaContractError(f"cuobjdump failed: {detail}")
    return output


def parse_listed_architectures(output: str) -> frozenset[str]:
    """Return architecture suffixes from cuobjdump list output."""

    return frozenset(match.lower() for match in _ARCHITECTURE_PATTERN.findall(output))


def inspect_device_code(
    cuobjdump: Path,
    backend: Path,
    variant: CudaVariant,
) -> dict[str, Any]:
    """Inspect and strictly match the PTX/SASS fatbin contract."""

    version_output = _run_cuobjdump(cuobjdump, "--version").strip()
    if CUOBJDUMP_VERSION not in version_output:
        raise CudaContractError(
            f"Expected cuobjdump {CUOBJDUMP_VERSION}, got {version_output!r}"
        )
    inspector_sha256 = file_sha256(cuobjdump)
    if inspector_sha256 != CUOBJDUMP_SHA256:
        raise CudaContractError(
            "cuobjdump executable digest differs from the pinned redistributable"
        )
    sass = parse_listed_architectures(
        _run_cuobjdump(cuobjdump, "--list-elf", str(backend))
    )
    ptx = parse_listed_architectures(
        _run_cuobjdump(cuobjdump, "--list-ptx", str(backend))
    )
    if sass != variant.sass_architectures:
        raise CudaContractError(
            f"CUDA {variant.cuda_version} SASS architectures differ: "
            f"expected {sorted(variant.sass_architectures)}, got {sorted(sass)}"
        )
    if ptx != variant.ptx_architectures:
        raise CudaContractError(
            f"CUDA {variant.cuda_version} PTX architectures differ: "
            f"expected {sorted(variant.ptx_architectures)}, got {sorted(ptx)}"
        )
    return {
        "inspector": {
            "name": "NVIDIA cuobjdump",
            "sha256": inspector_sha256,
            "version": version_output,
        },
        "ptx_architectures": sorted(ptx),
        "sass_architectures": sorted(sass),
    }


def validate_variant_metadata(manifest: Mapping[str, Any]) -> CudaVariant:
    """Validate compatibility and fatbin fields against the known variant."""

    cuda_version = manifest.get("cuda_version")
    variant = CUDA_VARIANTS.get(cuda_version)
    if variant is None:
        raise CudaContractError(f"Unsupported CUDA pack version: {cuda_version!r}")
    if manifest.get("cuda_major") != variant.cuda_major:
        raise CudaContractError(f"CUDA {cuda_version} major-version metadata differs")
    compatibility = manifest.get("compatibility")
    device_code = manifest.get("device_code")
    expected_compatibility = {
        "minimum_compute_capability": variant.minimum_compute_capability,
        "minimum_driver_family": variant.minimum_driver_family,
        "minimum_driver_api": variant.minimum_driver_api,
    }
    if compatibility != expected_compatibility:
        raise CudaContractError(
            f"CUDA {cuda_version} compatibility metadata differs from contract"
        )
    if not isinstance(device_code, Mapping):
        raise CudaContractError(f"CUDA {cuda_version} device-code metadata is missing")
    if set(device_code.get("ptx_architectures", [])) != set(
        variant.ptx_architectures
    ):
        raise CudaContractError(f"CUDA {cuda_version} PTX metadata differs")
    if set(device_code.get("sass_architectures", [])) != set(
        variant.sass_architectures
    ):
        raise CudaContractError(f"CUDA {cuda_version} SASS metadata differs")
    return variant


def select_cuda_pack(
    manifests: Iterable[Mapping[str, Any]],
    *,
    compute_capability: int,
    driver_family: int,
) -> Mapping[str, Any] | None:
    """Select the newest compatible pack from validated manifests."""

    compatible: list[tuple[int, Mapping[str, Any]]] = []
    for manifest in manifests:
        variant = validate_variant_metadata(manifest)
        if (
            compute_capability >= variant.minimum_compute_capability
            and driver_family >= variant.minimum_driver_family
        ):
            compatible.append((variant.cuda_major, manifest))
    if not compatible:
        return None
    return max(compatible, key=lambda item: item[0])[1]
