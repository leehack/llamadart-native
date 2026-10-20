#!/usr/bin/env python3
"""Check that a Windows bundle loads on a host without the Visual C++ runtime.

Every import of every DLL must be another DLL of the bundle or a library
Windows (or a GPU driver) provides, every named import from the bundle must
be exported there, and the bundled Visual C++ runtime must be at least as new
as the MSVC linker of every DLL that imports it.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess

import windows_runtime


def bundle_errors(images: dict[str, windows_runtime.PeImage], arch: str) -> list[str]:
    errors: list[str] = []
    for name, image in sorted(images.items()):
        if image.machine != arch:
            errors.append(f"{name}: {image.machine} image in a {arch} bundle")
        if windows_runtime.is_os_provided(name):
            errors.append(f"{name}: Windows provides this library; the bundle must not ship it")
        for dependency, symbols in sorted(image.all_imports().items()):
            if dependency in images:
                missing = sorted(s for s in symbols if not s.startswith("#") and s not in images[dependency].exports)
                if missing:
                    errors.append(f"{name}: {dependency} does not export {', '.join(missing[:5])}")
            elif not windows_runtime.is_os_provided(dependency):
                errors.append(f"{name}: imports {dependency}, which neither the bundle nor Windows provides")
    errors.extend(runtime_version_errors(images))
    return errors


def runtime_version_errors(images: dict[str, windows_runtime.PeImage]) -> list[str]:
    runtime = {name: image.file_version for name, image in images.items() if windows_runtime.is_vc_runtime(name)}
    if not runtime:
        return []
    errors = [f"{name}: no file version resource" for name, version in sorted(runtime.items()) if version is None]
    if errors:
        return errors
    if len(set(runtime.values())) > 1:
        listed = ", ".join(f"{name} {format_version(version)}" for name, version in sorted(runtime.items()))
        return [f"Visual C++ runtime DLLs come from different redistributables: {listed}"]
    version = next(iter(runtime.values()))
    errors = []
    for name, image in sorted(images.items()):
        if not any(windows_runtime.is_vc_runtime(dependency) for dependency in image.all_imports()):
            continue
        # Major version 14 is the MSVC v14 toolset family; lld-link writes 14.0.
        if image.linker_version[0] == 14 and image.linker_version > version[:2]:
            errors.append(
                f"{name}: linked by MSVC {format_version(image.linker_version)}, newer than the bundled "
                f"Visual C++ runtime {format_version(version)}"
            )
    return errors


def dumpbin_errors(dumpbin: str, directory: Path, images: dict[str, windows_runtime.PeImage]) -> list[str]:
    errors = []
    for name, image in sorted(images.items()):
        output = subprocess.run(
            [dumpbin, "/nologo", "/dependents", str(directory / name)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        # Dependency lines hold one bare file name; "Dump of file <path>" has spaces.
        listed = {
            line.strip().lower()
            for line in output.splitlines()
            if line.strip().lower().endswith(".dll") and " " not in line.strip()
        }
        parsed = set(image.all_imports())
        if listed != parsed:
            errors.append(
                f"{name}: dumpbin lists {sorted(listed - parsed)} and the PE reader {sorted(parsed - listed)} "
                "beyond each other"
            )
    return errors


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def report(directory: Path, images: dict[str, windows_runtime.PeImage]) -> None:
    for name, image in sorted(images.items()):
        bundled = sorted(d for d in image.all_imports() if not windows_runtime.is_os_provided(d))
        provided = sorted(d for d in image.all_imports() if windows_runtime.is_os_provided(d) and not d.startswith("api-ms-win-"))
        line = f"{name}: {image.machine}, linker {format_version(image.linker_version)}"
        if windows_runtime.is_vc_redistributable(name):
            digest = hashlib.sha256((directory / name).read_bytes()).hexdigest()
            version = format_version(image.file_version) if image.file_version else "unversioned"
            line += f", file version {version}, sha256 {digest}"
        print(line)
        print(f"  bundle: {', '.join(bundled) or '-'}")
        print(f"  Windows: {', '.join(provided) or '-'} (+ api-ms-win-* sets)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--arch", required=True, choices=("x64", "arm64"))
    parser.add_argument("--dumpbin", help="cross-check every import list against dumpbin /dependents")
    args = parser.parse_args()

    dlls = sorted(args.bundle.glob("*.dll"))
    if not dlls:
        parser.error(f"no DLLs in {args.bundle}")
    try:
        images = {dll.name.lower(): windows_runtime.read_pe(dll) for dll in dlls}
    except windows_runtime.PeError as error:
        parser.error(str(error))
    report(args.bundle, images)
    errors = bundle_errors(images, args.arch)
    if args.dumpbin:
        errors.extend(dumpbin_errors(args.dumpbin, args.bundle, images))
    if errors:
        print("\n".join(errors))
        return 1
    print(f"{args.bundle}: {len(images)} DLLs, import closure complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
