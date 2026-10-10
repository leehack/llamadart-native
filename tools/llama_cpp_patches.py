#!/usr/bin/env python3
"""Applies patches/llama.cpp to copies of upstream sources; see its README.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PATCHES_DIR = ROOT / "patches" / "llama.cpp"
SERIES_NAME = "series.json"
# A patched file replaces a source of an upstream target, so it must be one:
# a header is included from the submodule by sources this does not see.
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cxx", ".m", ".mm"})
PATCH_NAME = re.compile(r"\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.patch")
HUNK_HEADER = re.compile(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
SERIES_KEYS = frozenset({"file", "tracking", "upstream", "android_markers"})
MARKER_KEYS = frozenset({"library", "text"})


class PatchError(Exception):
    pass


class NotApplicable(PatchError):
    """The patch is well formed but does not fit this upstream source."""


@dataclass(frozen=True)
class Hunk:
    header: str
    before: str
    after: str


@dataclass(frozen=True)
class Patch:
    name: str
    path: str
    hunks: tuple[Hunk, ...]


def _read_text(path: Path) -> str:
    # Universal newlines: a Windows checkout may hold either side as CRLF.
    with path.open(encoding="utf-8", newline=None) as handle:
        return handle.read()


def load_series(patches_dir: Path = PATCHES_DIR) -> list[dict]:
    """Returns the entries of series.json, in the order they are applied."""
    series_path = patches_dir / SERIES_NAME
    try:
        entries = json.loads(series_path.read_text(encoding="utf-8"))["patches"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PatchError(f"{series_path}: expected {{\"patches\": [...]}}: {error}") from error
    if not isinstance(entries, list):
        raise PatchError(f"{series_path}: \"patches\" must be a list")
    names: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("file"), str):
            raise PatchError(f"{series_path}: every patch needs a \"file\"")
        name = entry["file"]
        unknown = sorted(set(entry) - SERIES_KEYS)
        if unknown:
            raise PatchError(f"{series_path}: {name}: unknown keys {unknown}")
        if PATCH_NAME.fullmatch(name) is None:
            raise PatchError(f"{series_path}: {name}: expected NNNN-lower-case-words.patch")
        for key in ("tracking", "upstream"):
            if not str(entry.get(key, "")).startswith("https://"):
                raise PatchError(f"{series_path}: {name}: \"{key}\" must be an https URL")
        # The artifact check is what keeps a release from losing a patch, so
        # every patch must give it something to find.
        markers = entry.get("android_markers")
        if not isinstance(markers, list) or not markers or not all(
            isinstance(marker, dict)
            and set(marker) == MARKER_KEYS
            and all(isinstance(value, str) and value for value in marker.values())
            for marker in markers
        ):
            raise PatchError(
                f"{series_path}: {name}: \"android_markers\" must list at least one "
                "{library, text}"
            )
        names.append(name)
    if names != sorted(set(names)):
        raise PatchError(f"{series_path}: patches must be unique and in name order")
    on_disk = sorted(path.name for path in patches_dir.glob("*.patch"))
    if on_disk != names:
        raise PatchError(
            f"{series_path}: lists {names} but {patches_dir} holds {on_disk}"
        )
    return entries


def parse_patch(name: str, text: str) -> Patch:
    """Parses a unified diff of one existing source file."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    paths: list[str] = []
    hunks: list[Hunk] = []
    header = ""
    before: list[str] = []
    after: list[str] = []

    def close_hunk() -> None:
        if header:
            if before == after:
                raise PatchError(f"{name}: {header}: hunk changes nothing")
            hunks.append(Hunk(header, "".join(before), "".join(after)))

    for line in lines:
        if not header:
            if line.startswith(("new file mode", "deleted file mode", "rename ", "Binary files")):
                raise PatchError(f"{name}: only edits of an existing text source are supported")
            if line.startswith("diff --git ") and paths:
                raise PatchError(f"{name}: a patch edits exactly one file")
            if line.startswith(("--- ", "+++ ")):
                prefix = "a/" if line.startswith("--- ") else "b/"
                if not line[4:].startswith(prefix):
                    raise PatchError(f"{name}: expected a -p1 path in {line!r}")
                paths.append(line[4 + len(prefix):])
                continue
        if HUNK_HEADER.match(line):
            close_hunk()
            header, before, after = line, [], []
        elif header:
            if line.startswith(" ") or line == "":
                before.append(line[1:] + "\n")
                after.append(line[1:] + "\n")
            elif line.startswith("-"):
                before.append(line[1:] + "\n")
            elif line.startswith("+"):
                after.append(line[1:] + "\n")
            else:
                raise PatchError(f"{name}: unsupported line {line!r}")
    close_hunk()
    if len(paths) != 2 or paths[0] != paths[1]:
        raise PatchError(f"{name}: expected one '--- a/PATH' and '+++ b/PATH' pair")
    path = PurePosixPath(paths[0])
    if path.is_absolute() or ".." in path.parts or path.suffix not in SOURCE_SUFFIXES:
        raise PatchError(
            f"{name}: {paths[0]} is not a compiled source inside llama.cpp "
            f"({', '.join(sorted(SOURCE_SUFFIXES))})"
        )
    if not hunks:
        raise PatchError(f"{name}: no hunks")
    return Patch(name, paths[0], tuple(hunks))


def apply_available(
    upstream: Path, patches_dir: Path = PATCHES_DIR
) -> tuple[dict[str, str], list[str]]:
    """Returns the patched text of each edited source, and why a patch was skipped.

    A hunk applies only where its context and removed lines occur exactly
    once: there is no fuzz and no offset search. A patch that does not fit
    this upstream is skipped whole.
    """
    patched: dict[str, str] = {}
    skipped: list[str] = []
    for entry in load_series(patches_dir):
        name = entry["file"]
        patch = parse_patch(name, _read_text(patches_dir / name))
        try:
            text = patched.get(patch.path)
            if text is None:
                source = upstream / patch.path
                if not source.is_file():
                    raise NotApplicable(f"{name}: {source} does not exist")
                text = _read_text(source)
            for hunk in patch.hunks:
                count = text.count(hunk.before)
                if count != 1:
                    raise NotApplicable(
                        f"{name}: {hunk.header} matches {patch.path} {count} times, expected 1"
                    )
                text = text.replace(hunk.before, hunk.after)
        except NotApplicable as error:
            skipped.append(str(error))
            continue
        patched[patch.path] = text
    return patched, skipped


def apply_series(upstream: Path, patches_dir: Path = PATCHES_DIR) -> dict[str, str]:
    """Like apply_available, but a patch that does not fit is an error."""
    patched, skipped = apply_available(upstream, patches_dir)
    if skipped:
        raise NotApplicable("\n".join(skipped))
    return patched


def manifest_entries(patches_dir: Path = PATCHES_DIR) -> list[dict[str, str]]:
    """What a release manifest records of the carried patches."""
    return [
        {
            "file": entry["file"],
            "sha256": hashlib.sha256((patches_dir / entry["file"]).read_bytes()).hexdigest(),
            "tracking": entry["tracking"],
        }
        for entry in load_series(patches_dir)
    ]


def android_marker_errors(out_dir: Path, patches_dir: Path = PATCHES_DIR) -> list[str]:
    """Names each bundled library that lacks the text a carried patch compiles in."""
    errors: list[str] = []
    for entry in load_series(patches_dir):
        for marker in entry["android_markers"]:
            library = out_dir / marker["library"]
            if library.is_file() and marker["text"].encode() not in library.read_bytes():
                errors.append(
                    f"{marker['library']}: built without {entry['file']} "
                    f"(no {marker['text']!r})"
                )
    return errors


def write_patched(
    upstream: Path, output: Path, patches_dir: Path, strict: bool = False
) -> tuple[list[str], list[str]]:
    """Writes the patched copies; returns their paths and the skipped patches."""
    if strict:
        patched, skipped = apply_series(upstream, patches_dir), []
    else:
        patched, skipped = apply_available(upstream, patches_dir)
    for relative, text in patched.items():
        target = output / relative
        # Keep the timestamp of an unchanged copy so reconfiguring rebuilds nothing.
        if target.is_file() and _read_text(target) == text:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return sorted(patched), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", type=Path, default=PATCHES_DIR)
    commands = parser.add_subparsers(dest="command", required=True)
    apply = commands.add_parser("apply", help="write patched copies and print their paths")
    apply.add_argument("--upstream", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--strict", action="store_true",
                       help="fail when a patch does not apply instead of skipping it")
    commands.add_parser("manifest", help="print the carried patches as JSON")
    args = parser.parse_args()
    try:
        if args.command == "apply":
            paths, skipped = write_patched(
                args.upstream, args.output, args.patches, args.strict)
            for reason in skipped:
                print(f"skipped {reason}", file=sys.stderr)
            print(";".join(paths))
        else:
            print(json.dumps(manifest_entries(args.patches)))
    except PatchError as error:
        print(f"llama.cpp patches: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
