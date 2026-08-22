#!/usr/bin/env python3
"""Package Linux runtime libraries without losing SONAME symlinks."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath, PureWindowsPath
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

    def add_member(path: Path, chain: tuple[Path, ...] = ()) -> None:
        if path in chain:
            names = " -> ".join(item.name for item in (*chain, path))
            raise ValueError(f"Linux runtime symlink cycle detected: {names}")

        if path.is_symlink():
            target = path.readlink()
            if (
                target.is_absolute()
                or len(target.parts) != 1
                or target.name in {".", ".."}
            ):
                raise ValueError(
                    f"Linux runtime symlink {path.name} must target a sibling file, "
                    f"not {target}"
                )

            target_path = path.parent / target
            if not (target_path.is_symlink() or target_path.is_file()):
                raise ValueError(
                    f"Linux runtime symlink target does not exist: "
                    f"{path.name} -> {target}"
                )

            selected[path.name] = path
            add_member(target_path, (*chain, path))
            return

        if path.is_file():
            selected[path.name] = path

    selected_patterns = patterns or ["*.so", "*.so.*"]
    for pattern in selected_patterns:
        posix_pattern = PurePosixPath(pattern)
        windows_pattern = PureWindowsPath(pattern)
        if (
            not pattern
            or posix_pattern.is_absolute()
            or windows_pattern.is_absolute()
            or windows_pattern.drive
            or "/" in pattern
            or "\\" in pattern
            or len(posix_pattern.parts) != 1
            or len(windows_pattern.parts) != 1
            or pattern in {".", ".."}
        ):
            raise ValueError(
                f"Linux runtime pattern must be a filename glob without path "
                f"components: {pattern!r}"
            )
        for path in input_dir.glob(pattern):
            add_member(path)

    if not selected:
        raise ValueError(
            f"No Linux runtime libraries matched {selected_patterns} "
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
