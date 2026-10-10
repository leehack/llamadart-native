#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import llama_cpp_patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--readelf", required=True, type=Path)
    parser.add_argument("--min-load-align", type=lambda value: int(value, 0), default=0x4000)
    return parser.parse_args()


def read_load_alignments(readelf: Path, library: Path) -> list[int]:
    result = subprocess.run(
        [str(readelf), "-lW", str(library)],
        capture_output=True,
        text=True,
        check=True,
    )
    alignments: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("LOAD"):
            continue
        fields = stripped.split()
        try:
            alignments.append(int(fields[-1], 16))
        except (IndexError, ValueError):
            continue
    return alignments


# The C++ exception runtime that a throw in one library and a typed catch in
# another must share. The NDK links it statically (ANDROID_STL=c++_static), so
# it works only while exactly one library of the bundle defines these and the
# others import them from it.
EXCEPTION_RUNTIME_SYMBOLS = (
    "__cxa_throw",
    "__cxa_begin_catch",
    "_ZTISt9exception",
    "_ZTISt13runtime_error",
)
# What the two ends of the exception barrier use: llama.cpp throws
# std::runtime_error, and libllamadart catches std::exception. A library with
# a private copy of the runtime would import neither.
REQUIRED_EXCEPTION_RUNTIME_IMPORTS = {
    "libllama.so": ("__cxa_throw", "_ZTISt13runtime_error"),
    "libllamadart.so": ("__cxa_begin_catch", "_ZTISt9exception"),
}


def read_dynamic_symbols(readelf: Path, library: Path) -> tuple[set[str], set[str]]:
    """Returns the names that library defines and those it imports."""
    result = subprocess.run(
        [str(readelf), "--dyn-syms", "-W", str(library)],
        capture_output=True,
        text=True,
        check=True,
    )
    defined: set[str] = set()
    imported: set[str] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].rstrip(":").isdigit():
            continue
        name = fields[7].split("@", 1)[0]
        (imported if fields[6] == "UND" else defined).add(name)
    return defined, imported


def exception_runtime_errors(symbols: dict[str, tuple[set[str], set[str]]]) -> list[str]:
    errors: list[str] = []
    for symbol in EXCEPTION_RUNTIME_SYMBOLS:
        definers = sorted(name for name, (defined, _) in symbols.items() if symbol in defined)
        importers = sorted(name for name, (_, imported) in symbols.items() if symbol in imported)
        if len(definers) > 1:
            errors.append(
                f"{symbol}: defined by {', '.join(definers)}; an exception thrown "
                "with one copy of the C++ runtime is not caught by type with another"
            )
        elif not definers and importers:
            errors.append(
                f"{symbol}: imported by {', '.join(importers)} but defined by no library of the bundle"
            )
    for library, required in REQUIRED_EXCEPTION_RUNTIME_IMPORTS.items():
        if library not in symbols:
            continue
        _, imported = symbols[library]
        for symbol in required:
            if symbol not in imported:
                errors.append(
                    f"{library}: does not import {symbol}, so it does not share "
                    "the C++ exception runtime of the bundle"
                )
    return errors


def main() -> None:
    args = parse_args()
    if not args.readelf.is_file():
        raise SystemExit(f"Missing llvm-readelf binary: {args.readelf}")
    if not args.out_dir.is_dir():
        raise SystemExit(f"Missing Android artifact directory: {args.out_dir}")

    libraries = sorted(args.out_dir.glob("*.so"))
    if not libraries:
        raise SystemExit(f"No shared libraries found under {args.out_dir}")

    errors: list[str] = []
    symbols = {library.name: read_dynamic_symbols(args.readelf, library) for library in libraries}
    errors.extend(exception_runtime_errors(symbols))
    errors.extend(llama_cpp_patches.android_marker_errors(args.out_dir))
    for symbol in EXCEPTION_RUNTIME_SYMBOLS:
        for name, (defined, _) in sorted(symbols.items()):
            if symbol in defined:
                print(f"{symbol}: defined by {name}")
    for library in libraries:
        alignments = read_load_alignments(args.readelf, library)
        if not alignments:
            errors.append(f"{library.name}: no LOAD program headers found")
            continue

        formatted = ", ".join(f"0x{alignment:x}" for alignment in alignments)
        print(f"{library.name}: LOAD alignments {formatted}")
        if any(alignment < args.min_load_align for alignment in alignments):
            errors.append(
                f"{library.name}: expected all LOAD alignments >= 0x{args.min_load_align:x}, got {formatted}"
            )

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
