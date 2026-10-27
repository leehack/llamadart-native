#!/usr/bin/env python3
"""Checks that a built library carries a complete build identity."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

KEYS = ("owner", "llama.cpp", "ggml", "patches", "config",
        "compiler", "target", "backends", "options")
# Read from the file's bytes so that libraries built for another platform can
# be checked on any runner.
IDENTITY = re.compile(rb"llamadart-native-build/1;[A-Za-z0-9._+,;=-]*")
DIRTY = "+dirty."


def identities(path: Path) -> set[str]:
    return {match.decode() for match in IDENTITY.findall(path.read_bytes())}


def check_value(path: Path, value: str) -> list[str]:
    fields = value.split(";")[1:]
    keys = [field.split("=", 1)[0] for field in fields]
    if keys != list(KEYS) or any(field.endswith("=") for field in fields):
        return [f"{path}: incomplete build identity: {value}"]
    print(f"{path}: {value}")
    # A release whose llama.cpp tag differs from the pinned gitlink is dirty
    # by construction, so only the stamp's presence is required.
    if DIRTY in value:
        print(f"{path}: note: built from a tree with tracked changes")
    return []


def check(path: Path) -> list[str]:
    found = sorted(identities(path))
    if not found:
        return [f"{path}: no build identity"]
    return [error for value in found for error in check_value(path, value)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("libraries", nargs="+", type=Path)
    errors = [error for path in parser.parse_args().libraries for error in check(path)]
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
