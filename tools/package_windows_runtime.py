#!/usr/bin/env python3
"""Stage lane redistributables and install one runtime set per Windows release bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import windows_runtime


RUNTIME_DIR = "vc-runtime"
VERSION_DLL = "msvcp140.dll"


def stage(payload: Path, arch: str, redist_dir: Path) -> None:
    files = windows_runtime.vc_redistributables(arch, redist_dir)
    if VERSION_DLL not in files:
        raise ValueError(f"Missing {arch} {VERSION_DLL} in VCToolsRedistDir: {redist_dir}")

    runtime = payload / RUNTIME_DIR
    runtime.mkdir()
    for name, source in files.items():
        shutil.copy2(source, runtime / name)

    # The release merge copies top-level DLLs in artifact name order.
    for dll in payload.glob("*.dll"):
        if windows_runtime.is_vc_redistributable(dll.name):
            dll.unlink()


def runtime_version(runtime: Path, arch: str) -> tuple[int, int, int, int]:
    dll = runtime / VERSION_DLL
    image = windows_runtime.read_pe(dll)
    if image.machine != arch or image.file_version is None:
        raise ValueError(f"Expected a versioned {arch} runtime DLL at {dll}")
    return image.file_version


def install(bundle: Path, arch: str, artifacts: Path) -> None:
    candidates = sorted(lane / RUNTIME_DIR for lane in artifacts.glob(f"native_windows_{arch}_*") if lane.is_dir())
    if not candidates:
        raise ValueError(f"No Windows {arch} lane artifacts in {artifacts}")
    runtime = max(candidates, key=lambda candidate: runtime_version(candidate, arch))
    print(f"{bundle}: using Visual C++ runtime from {runtime.parent.name}")

    # A backend or one of the selected runtime DLLs can import a DLL the core lane never needed.
    pending = list(bundle.glob("*.dll"))
    required: dict[str, Path] = {}
    while pending:
        for name in windows_runtime.read_pe(pending.pop()).all_imports():
            if not windows_runtime.is_vc_redistributable(name) or name in required:
                continue
            source = runtime / name
            if not source.is_file():
                raise ValueError(f"{runtime}: missing required redistributable {name}")
            required[name] = source
            pending.append(source)

    for name, source in sorted(required.items()):
        shutil.copy2(source, bundle / name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    stage_parser = commands.add_parser("stage")
    stage_parser.add_argument("payload", type=Path)
    install_parser = commands.add_parser("install")
    install_parser.add_argument("bundle", type=Path)
    install_parser.add_argument("artifacts", type=Path)
    for command in (stage_parser, install_parser):
        command.add_argument("--arch", required=True, choices=("x64", "arm64"))

    args = parser.parse_args()
    try:
        if args.command == "stage":
            redist_dir = os.environ.get("VCToolsRedistDir")
            if not redist_dir:
                stage_parser.error("VCToolsRedistDir must point to the lane's MSVC redistributables")
            stage(args.payload, args.arch, Path(redist_dir))
        else:
            install(args.bundle, args.arch, args.artifacts)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
