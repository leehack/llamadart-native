#!/usr/bin/env python3
"""Writes the Metal kernel sources embedded in a libllamadart.dylib to a directory."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

SYMBOL = re.compile(r"^([0-9a-f]+) \S _ggml_metallib_(\w+)_(start|end)$", re.M)
SECTION = re.compile(
    r"sectname __ggml_metallib\n\s+segname \S+\n\s+addr 0x([0-9a-f]+)\n\s+size \S+\n\s+offset (\d+)"
)


def extract(dylib: Path, output: Path) -> list[str]:
    symbols = subprocess.check_output(["nm", "-gU", str(dylib)], text=True)
    bounds: dict[str, dict[str, int]] = {}
    for address, kind, edge in SYMBOL.findall(symbols):
        bounds.setdefault(kind, {})[edge] = int(address, 16)
    section = SECTION.search(subprocess.check_output(["otool", "-l", str(dylib)], text=True))
    if section is None or not bounds:
        raise ValueError(f"{dylib} embeds no Metal kernels")
    file_offset = int(section[2]) - int(section[1], 16)
    data = dylib.read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    for kind, edges in bounds.items():
        start, end = file_offset + edges["start"], file_offset + edges["end"]
        (output / f"{kind}.metal").write_bytes(data[start:end])
    return sorted(bounds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dylib", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        kinds = extract(args.dylib, args.output)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"extract_metal_kernels: {error}", file=sys.stderr)
        return 1
    print(f"{len(kinds)} kernel sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
