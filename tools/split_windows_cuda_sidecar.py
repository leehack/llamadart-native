#!/usr/bin/env python3
"""Move the CUDA backend of the merged windows-x64 bundle into its own sidecar.

The sidecar holds ggml-cuda.dll and every DLL only it needs (cudart, cuBLAS,
cuBLASLt), so the windows-x64 core archive stays small for apps that never
select CUDA. The split is derived from the PE import tables and must equal
SIDECAR_DLLS exactly: a CUDA major bump or a new CUDA dependency changes the
sidecar's name or contents and has to be made on purpose. Both the merged
bundle and the core left behind must pass the import-closure check.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import validate_windows_bundle
import windows_runtime

ARCH = "x64"
CUDA_BACKEND = "ggml-cuda.dll"
SIDECAR_DLLS = frozenset({CUDA_BACKEND, "cudart64_12.dll", "cublas64_12.dll", "cublaslt64_12.dll"})


def _reachable(roots: set[str], images: dict[str, windows_runtime.PeImage]) -> set[str]:
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in seen or name not in images:
            continue
        seen.add(name)
        pending.extend(images[name].all_imports())
    return seen


def sidecar_members(images: dict[str, windows_runtime.PeImage]) -> set[str]:
    """CUDA backend plus the bundle DLLs nothing outside it imports."""
    if CUDA_BACKEND not in images:
        raise ValueError(f"{CUDA_BACKEND} is missing from the windows-{ARCH} bundle")
    cuda = _reachable({CUDA_BACKEND}, images)
    core_roots = set(images) - cuda
    return cuda - _reachable(core_roots, images)


def split(bundle: Path, sidecar: Path) -> list[str]:
    files = {path.name.lower(): path for path in sorted(bundle.glob("*.dll"))}
    images = {name: windows_runtime.read_pe(path) for name, path in files.items()}
    errors = [f"merged bundle: {error}" for error in validate_windows_bundle.bundle_errors(images, ARCH)]
    if errors:
        return errors
    members = sidecar_members(images)
    if members != SIDECAR_DLLS:
        return [
            f"CUDA sidecar would hold {sorted(members)}, expected {sorted(SIDECAR_DLLS)}; "
            "update SIDECAR_DLLS and the sidecar asset name together"
        ]
    if sidecar.exists() and any(sidecar.iterdir()):
        return [f"{sidecar} is not empty"]
    sidecar.mkdir(parents=True, exist_ok=True)
    for name in sorted(members):
        files[name].rename(sidecar / files[name].name)
    core = {name: image for name, image in images.items() if name not in members}
    return [f"core bundle: {error}" for error in validate_windows_bundle.bundle_errors(core, ARCH)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="merged windows-x64 bundle directory, split in place")
    parser.add_argument("sidecar", type=Path, help="directory that receives the CUDA sidecar")
    args = parser.parse_args()
    try:
        errors = split(args.bundle, args.sidecar)
    except ValueError as error:
        errors = [str(error)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"{args.sidecar}: {', '.join(sorted(SIDECAR_DLLS))}")
    print(f"{args.bundle}: core import closure complete without CUDA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
