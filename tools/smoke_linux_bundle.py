#!/usr/bin/env python3
"""Load a packaged Linux x64 wrapper and call its model-free version probe."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def smoke(bundle: Path) -> None:
    bundle = bundle.resolve()
    wrapper = bundle / "libllamadart.so"
    if not wrapper.is_file():
        raise RuntimeError(f"missing packaged wrapper: {wrapper}")

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = str(bundle)
    probe = (
        "import ctypes,sys; "
        "lib=ctypes.CDLL(sys.argv[1]); "
        "lib.llama_dart_tts_api_version.restype=ctypes.c_uint32; "
        "version=lib.llama_dart_tts_api_version();\n"
        "if version != 1:\n"
        "    raise RuntimeError(f'unexpected wrapper API version: {version}')\n"
        "print(f'loaded libllamadart.so; TTS API version={version}')"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe, str(wrapper)],
            capture_output=True,
            text=True,
            env=environment,
            timeout=60,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("timed out loading packaged wrapper") from error
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "load probe failed").strip()
        raise RuntimeError(detail)
    print(result.stdout.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        smoke(args.bundle)
    except RuntimeError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
