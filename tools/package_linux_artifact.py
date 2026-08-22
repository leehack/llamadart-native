#!/usr/bin/env python3
"""Package Linux runtime libraries without losing SONAME symlinks."""

from __future__ import annotations

import argparse
from pathlib import Path
import tarfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Filename glob to include (repeatable; default: *.so and *.so.*)",
    )
    return parser.parse_args()


def select_members(input_dir: Path, patterns: list[str]) -> list[Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    selected: dict[str, Path] = {}
    for pattern in patterns or ["*.so", "*.so.*"]:
        for path in input_dir.glob(pattern):
            if path.is_file():
                selected[path.name] = path

    if not selected:
        raise ValueError(
            f"No Linux runtime libraries matched {patterns or ['*.so', '*.so.*']} "
            f"under {input_dir}"
        )
    return [selected[name] for name in sorted(selected)]


def main() -> int:
    args = parse_args()
    try:
        members = select_members(args.input_dir, args.pattern)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w:gz", dereference=False) as archive:
        for path in members:
            archive.add(path, arcname=path.name, recursive=False)

    print(f"Packaged {len(members)} Linux runtime entries in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
