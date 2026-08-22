#!/usr/bin/env python3
"""Validate Linux runtime archive SONAME and dependency contracts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile


LOCAL_LIBRARY_PREFIXES = ("libllamadart", "libllama", "libggml", "libmtmd")
PLACEHOLDER = "SOVERSION"


@dataclass(frozen=True)
class DynamicMetadata:
    soname: str | None
    needed: tuple[str, ...]
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--tool",
        help="ELF inspection tool (default: first available readelf, llvm-readelf, or objdump)",
    )
    return parser.parse_args()


def resolve_tool(explicit: str | None) -> tuple[str, str]:
    if explicit:
        resolved = shutil.which(explicit) or explicit
        mode = "objdump" if Path(resolved).name.endswith("objdump") else "readelf"
        return resolved, mode
    for candidate, mode in (
        ("readelf", "readelf"),
        ("llvm-readelf", "readelf"),
        ("objdump", "objdump"),
    ):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved, mode
    raise ValueError("No ELF inspection tool found (tried readelf, llvm-readelf, and objdump)")


def inspect_dynamic(path: Path, tool: str, mode: str) -> DynamicMetadata:
    command = [tool, "-p" if mode == "objdump" else "-d", str(path)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    raw = result.stdout
    soname: str | None = None
    needed: list[str] = []
    for line in raw.splitlines():
        readelf_match = re.search(r"\((NEEDED|SONAME)\).*\[([^]]+)\]", line)
        objdump_match = re.match(r"\s*(NEEDED|SONAME)\s+(\S+)", line)
        match = readelf_match or objdump_match
        if not match:
            continue
        kind, value = match.groups()
        if kind == "SONAME":
            soname = value
        else:
            needed.append(value)
    return DynamicMetadata(soname=soname, needed=tuple(needed), raw=raw)


def safe_member_name(name: str) -> str:
    normalized = name.removeprefix("./")
    path = PurePosixPath(normalized)
    if path.is_absolute() or len(path.parts) != 1 or path.name in ("", ".", ".."):
        raise ValueError(f"Archive member must be a flat runtime filename: {name}")
    return path.name


def extract_archive(archive_path: Path, destination: Path) -> dict[str, tarfile.TarInfo]:
    members: dict[str, tarfile.TarInfo] = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            name = safe_member_name(member.name)
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            if not (member.isfile() or member.issym()):
                raise ValueError(f"Unsupported archive member type: {name}")
            if member.issym():
                target = safe_member_name(member.linkname)
                if target == name:
                    raise ValueError(f"Self-referential archive symlink: {name}")
            members[name] = member
            # Every member and symlink target is constrained to one flat
            # filename above, so extraction cannot escape the temporary root.
            archive.extract(member, destination)
    return members


def validate_symlinks(members: dict[str, tarfile.TarInfo], errors: list[str]) -> None:
    for name, member in members.items():
        if not member.issym():
            continue
        target = safe_member_name(member.linkname)
        if target not in members:
            errors.append(f"{name}: symlink target is absent from archive: {target}")


def resolve_symlink_member(
    name: str, members: dict[str, tarfile.TarInfo]
) -> str | None:
    visited: set[str] = set()
    while name in members and members[name].issym():
        if name in visited:
            return None
        visited.add(name)
        name = safe_member_name(members[name].linkname)
    return name if name in members else None


def validate_archive(archive_path: Path, tool: str, mode: str) -> list[str]:
    errors: list[str] = []
    if not archive_path.is_file():
        return [f"Archive does not exist: {archive_path}"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        try:
            members = extract_archive(archive_path, root)
        except (ValueError, tarfile.TarError) as error:
            return [str(error)]

        validate_symlinks(members, errors)
        names = set(members)
        for required in ("libllamadart.so", "libmtmd.so"):
            if required not in names:
                errors.append(f"Missing required archive member: {required}")

        metadata: dict[str, DynamicMetadata] = {}
        for name in sorted(names):
            if ".so" not in name or members[name].issym():
                continue
            try:
                dynamic = inspect_dynamic(root / name, tool, mode)
            except subprocess.CalledProcessError as error:
                errors.append(f"{name}: ELF inspection failed with exit code {error.returncode}")
                continue
            metadata[name] = dynamic
            if PLACEHOLDER in name or PLACEHOLDER in dynamic.raw:
                errors.append(f"{name}: contains literal {PLACEHOLDER} in filename or dynamic metadata")
            if dynamic.soname and dynamic.soname not in names:
                errors.append(f"{name}: SONAME {dynamic.soname} is absent from archive")
            for dependency in dynamic.needed:
                if dependency.startswith(LOCAL_LIBRARY_PREFIXES) and dependency not in names:
                    errors.append(f"{name}: local DT_NEEDED dependency is absent: {dependency}")

        mtmd = metadata.get("libmtmd.so")
        if mtmd is None:
            target = resolve_symlink_member("libmtmd.so", members)
            mtmd = metadata.get(target or "")
        if mtmd is None or not mtmd.soname:
            errors.append("libmtmd ELF does not declare a SONAME")
        elif not re.fullmatch(r"libmtmd\.so\.\d+", mtmd.soname):
            errors.append(f"libmtmd ELF has unresolved SONAME: {mtmd.soname}")

        canonical_mtmd = members.get("libmtmd.so")
        if canonical_mtmd is not None and not canonical_mtmd.issym():
            errors.append("libmtmd.so must remain a symlink to its versioned ELF")

    return errors


def main() -> int:
    args = parse_args()
    try:
        tool, mode = resolve_tool(args.tool)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1
    errors = validate_archive(args.archive, tool, mode)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Linux artifact contract verified: {args.archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
