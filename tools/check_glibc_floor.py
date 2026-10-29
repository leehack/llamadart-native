#!/usr/bin/env python3
"""Fail when a shipped Linux ``.so`` imports a GLIBC symbol version above the floor.

The floor is the highest ``GLIBC_x.y`` the pinned ``ubuntu-24.04`` release
build imports. Building on a newer image silently raises it (Ubuntu 26.04
imported ``sqrtf@GLIBC_2.43``), which stops the bundle loading on Ubuntu 24.04
and Debian 13. Raise ``GLIBC_FLOOR`` only with an announced consumer floor
raise (docs/platform_backend_strategy.md).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

GLIBC_FLOOR = (2, 38)
UNDEFINED_GLIBC_IMPORT = re.compile(r"\sUND\s+(\S+?)@GLIBC_(\d+(?:\.\d+)+)")


def parse_version(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def imports_above_floor(readelf_dyn_syms: str, floor: tuple[int, ...] = GLIBC_FLOOR) -> dict[str, tuple[int, ...]]:
    above: dict[str, tuple[int, ...]] = {}
    for line in readelf_dyn_syms.splitlines():
        match = UNDEFINED_GLIBC_IMPORT.search(line)
        if match and parse_version(match[2]) > floor:
            above[match[1]] = parse_version(match[2])
    return above


def library_paths(paths: list[Path]) -> list[Path]:
    libraries: list[Path] = []
    for path in paths:
        libraries.extend(sorted(path.glob("*.so*")) if path.is_dir() else [path])
    return libraries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--readelf", default=os.environ.get("READELF", "readelf"))
    parser.add_argument("--floor", default=".".join(map(str, GLIBC_FLOOR)))
    parser.add_argument("paths", nargs="+", type=Path, help="shared libraries or directories of them")
    args = parser.parse_args()
    floor = parse_version(args.floor)
    libraries = library_paths(args.paths)
    if not libraries:
        print("No shared libraries found to check", file=sys.stderr)
        return 1
    failed = False
    for library in libraries:
        try:
            symbols = subprocess.check_output([args.readelf, "--dyn-syms", "--wide", str(library)], text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"readelf failed on {library}: {error}", file=sys.stderr)
            return 1
        for name, version in sorted(imports_above_floor(symbols, floor).items()):
            failed = True
            print(
                f"{library}: {name}@GLIBC_{'.'.join(map(str, version))} is above the GLIBC_{args.floor} floor",
                file=sys.stderr,
            )
    if failed:
        return 1
    print(f"PASS {len(libraries)} libraries import no GLIBC symbol version above GLIBC_{args.floor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
