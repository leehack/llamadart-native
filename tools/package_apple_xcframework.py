#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "bin"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "release_assets"
DEFAULT_WORK_DIR = REPO_ROOT / ".dart_tool" / "apple_xcframework"
LIBRARY_NAME = "libllamadart.dylib"
MODULE_NAME = "llamadart_native"


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def resolve_library(input_dir: Path, nested: str, flat: str) -> Path:
    nested_path = input_dir / nested / LIBRARY_NAME
    if nested_path.is_file():
        return nested_path
    flat_path = input_dir / flat / LIBRARY_NAME
    if flat_path.is_file():
        return flat_path
    raise RuntimeError(
        f"Missing required file: expected {nested_path} or {flat_path}"
    )


def make_universal_library(inputs: list[Path], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "xcrun",
            "lipo",
            "-create",
            *[str(path) for path in inputs],
            "-output",
            str(output),
        ]
    )
    return output


def copy_headers(headers_dir: Path) -> None:
    include_root = REPO_ROOT / "third_party" / "llama.cpp" / "include"
    ggml_include_root = REPO_ROOT / "third_party" / "llama.cpp" / "ggml" / "include"
    mtmd_root = REPO_ROOT / "third_party" / "llama.cpp" / "tools" / "mtmd"
    wrapper_header = REPO_ROOT / "src" / "llama_dart_wrapper.h"

    headers_dir.mkdir(parents=True, exist_ok=True)
    for source in [
        wrapper_header,
        include_root / "llama.h",
        ggml_include_root / "ggml.h",
        ggml_include_root / "ggml-alloc.h",
        ggml_include_root / "ggml-backend.h",
        ggml_include_root / "ggml-cpu.h",
        ggml_include_root / "ggml-opt.h",
        ggml_include_root / "gguf.h",
        mtmd_root / "mtmd.h",
        mtmd_root / "mtmd-helper.h",
    ]:
        if source.is_file():
            shutil.copy2(source, headers_dir / source.name)

    umbrella = headers_dir / f"{MODULE_NAME}.h"
    umbrella.write_text(
        "\n".join(
            [
                "#pragma once",
                '#include "llama_dart_wrapper.h"',
                '#include "llama.h"',
                '#include "ggml.h"',
                '#include "ggml-alloc.h"',
                '#include "ggml-backend.h"',
                '#include "ggml-cpu.h"',
                '#include "ggml-opt.h"',
                '#include "gguf.h"',
                '#include "mtmd.h"',
                '#include "mtmd-helper.h"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    (headers_dir / "module.modulemap").write_text(
        "\n".join(
            [
                f"module {MODULE_NAME} {{",
                f'  umbrella header "{MODULE_NAME}.h"',
                "  export *",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def zip_xcframework(xcframework: Path, output_zip: Path) -> None:
    if output_zip.exists():
        output_zip.unlink()
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(xcframework.rglob("*")):
            if path.is_dir():
                continue
            archive.write(path, path.relative_to(xcframework.parent))


def package_xcframework(
    input_dir: Path,
    output_dir: Path,
    work_dir: Path,
    tag: str,
    clean: bool,
) -> Path:
    if clean and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    ios_device = resolve_library(input_dir, "ios/arm64", "ios-arm64")
    ios_sim_arm64 = resolve_library(input_dir, "ios/arm64-sim", "ios-arm64-sim")
    ios_sim_x64 = resolve_library(input_dir, "ios/x86_64-sim", "ios-x86_64-sim")
    macos_arm64 = resolve_library(input_dir, "macos/arm64", "macos-arm64")
    macos_x64 = resolve_library(input_dir, "macos/x86_64", "macos-x86_64")

    ios_sim_universal = make_universal_library(
        [ios_sim_arm64, ios_sim_x64],
        work_dir / "ios-simulator" / LIBRARY_NAME,
    )
    macos_universal = make_universal_library(
        [macos_arm64, macos_x64],
        work_dir / "macos" / LIBRARY_NAME,
    )

    headers_dir = work_dir / "Headers"
    copy_headers(headers_dir)

    xcframework = work_dir / f"{MODULE_NAME}.xcframework"
    if xcframework.exists():
        shutil.rmtree(xcframework)
    run(
        [
            "xcodebuild",
            "-create-xcframework",
            "-library",
            str(ios_device),
            "-headers",
            str(headers_dir),
            "-library",
            str(ios_sim_universal),
            "-headers",
            str(headers_dir),
            "-library",
            str(macos_universal),
            "-headers",
            str(headers_dir),
            "-output",
            str(xcframework),
        ]
    )

    output_zip = output_dir / f"llamadart-native-apple-xcframework-{tag}.zip"
    zip_xcframework(xcframework, output_zip)
    print(f"Wrote {output_zip}", flush=True)
    return output_zip


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Package Apple libllamadart slices as an SPM-compatible XCFramework zip."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    package_xcframework(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        work_dir=args.work_dir.resolve(),
        tag=args.tag,
        clean=args.clean,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
