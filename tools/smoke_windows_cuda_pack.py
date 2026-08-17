#!/usr/bin/env python3
"""Smoke a repackaged CUDA backend through the exact Windows ggml loader."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path


LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required smoke file is missing: {path}")


def smoke(directory: Path, backend_name: str) -> dict[str, object]:
    if os.name != "nt":
        raise RuntimeError("Windows CUDA pack smoke must run on Windows")

    directory = directory.resolve()
    backend_path = directory / backend_name
    ggml_path = directory / "ggml.dll"
    for path in (backend_path, ggml_path, directory / "ggml-base.dll"):
        require_file(path)

    with os.add_dll_directory(str(directory)):
        backend_library = ctypes.CDLL(
            str(backend_path), winmode=LOAD_WITH_ALTERED_SEARCH_PATH
        )
        getattr(backend_library, "ggml_backend_init")

        ggml_library = ctypes.CDLL(
            str(ggml_path), winmode=LOAD_WITH_ALTERED_SEARCH_PATH
        )
        load = ggml_library.ggml_backend_load
        load.argtypes = [ctypes.c_char_p]
        load.restype = ctypes.c_void_p
        unload = ggml_library.ggml_backend_unload
        unload.argtypes = [ctypes.c_void_p]
        unload.restype = None

        registry = load(os.fsencode(backend_path))
        if not registry:
            raise RuntimeError(
                f"ggml_backend_load rejected the repackaged backend: {backend_path}"
            )
        unload(registry)

    return {
        "backend": backend_name,
        "directory": str(directory),
        "direct_load": True,
        "ggml_backend_load": True,
        "ggml_backend_unload": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--backend", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = smoke(args.directory, args.backend)
    except (AttributeError, OSError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
