# llamadart-native

Native build and release pipeline for `llamadart` binaries.

## Purpose

This repository is responsible for:

- Building native `llamadart` binaries across platforms.
- Publishing release artifacts consumed by `llamadart` build hooks.
- Producing release metadata (`assets.json` + `SHA256SUMS`).

The Dart API/runtime stays in the main `llamadart` repository.

## Workflow

- `Native Build & Release` (`.github/workflows/native_release.yml`)
  - Manual dispatch.
  - Builds one full backend set per platform/arch target.
  - Fails when any enabled backend in that target fails.
  - Publishes per-target native assets (Apple consolidated, others split core/backend libs).
  - Generates `assets.json` and `SHA256SUMS`.
- `Auto Trigger Native Release` (`.github/workflows/auto_native_release.yml`)
  - Daily schedule plus manual dispatch.
  - Resolves latest upstream `ggml-org/llama.cpp` release tag.
  - Dispatches `Native Build & Release` only when this repo does not already have that tag and no native release run is in flight.

## Backend Policy (Worthy Sets)

Each target builds all worthy backends together in one build:

- Android: arm64 = Vulkan + OpenCL + Kleidi + CPU; x86_64 = Vulkan + OpenCL + CPU
- iOS/macOS: Metal + CPU (consolidated into `libllamadart`, BLAS/Kleidi disabled)
- Linux x64: Vulkan + CUDA + BLAS + ZenDNN + CPU
- Linux arm64: Vulkan + BLAS + Kleidi + CPU
- Windows x64: Vulkan + CUDA + BLAS + CPU
- Windows arm64: Vulkan + BLAS + Kleidi + CPU

Non-Apple targets use `GGML_BACKEND_DL=ON`, so backend libs are optional at package/runtime level.

## Runtime Packaging Model

Release assets contain:

- Apple: consolidated `libllamadart` per target.
- Non-Apple core libs: `llamadart`, `llama`, `ggml`, `ggml-base` (and `mtmd` where produced)
- Non-Apple backend libs: `ggml-<backend>` modules (`ggml-vulkan`, `ggml-opencl`, etc.)
- Windows backend runtime deps:
  - CUDA lanes include CUDA runtime DLLs required by `ggml-cuda` (for example `cudart64_*.dll`, `cublas64_*.dll`).
  - BLAS lanes include `openblas*.dll` required by `ggml-blas`.
  - NVIDIA driver DLLs (for example `nvcuda.dll`) are not bundled and are provided by GPU drivers.
- Headers archive: `llamadart-native-headers-<tag>.tar.gz` with `llama_cpp/...` and `libllamadart/...` roots, including llama.cpp, ggml, mtmd, and `llama_dart_wrapper.h`.

Consumers can choose which backend libs to include in their package and load at runtime.

## Release Asset Naming

Assets are suffixed with platform/arch, for example:

- `libllamadart-linux-x64.so`
- `libllama-linux-x64.so`
- `libggml-vulkan-linux-x64.so`
- `libggml-opencl-android-arm64.so`
- `ggml-cuda-windows-x64.dll`

## Repository Layout

- `.github/workflows/auto_native_release.yml`: daily upstream tag watcher + native release dispatcher.
- `.github/workflows/native_release.yml`: build + package + release.
- `.gitmodules`: pinned native dependency submodules.
- `CMakeLists.txt` + `CMakePresets.json`: root-native build configuration.
- `src/`: `llama_dart_wrapper.*`.
- `third_party/llama.cpp`: upstream llama.cpp submodule.
- `third_party/Vulkan-Headers`: Vulkan headers submodule for Android Vulkan builds.
- `third_party/OpenCL-Headers`: OpenCL headers submodule (Android OpenCL builds).
- `third_party/OpenCL-ICD-Loader`: OpenCL loader submodule used to produce Android `libOpenCL.so` when NDK does not provide one.
- `third_party/opencl-stubs`: optional local fallback location for OpenCL headers/stubs.
- `tools/build.py`: cross-platform build entrypoint.
- `scripts/generate_assets_manifest.sh`: builds `assets.json` + checksums.
- `docs/platform_backend_strategy.md`: platform/backend matrix.

## Local Build (Preferred)

Builds are driven by root `CMakePresets.json` via `tools/build.py`.

Examples:

```bash
# macOS arm64 (Metal + CPU)
python3 tools/build.py apple --target macos-arm64

# Linux x64 (Vulkan + CUDA + BLAS + ZenDNN + CPU)
python3 tools/build.py linux --arch x64

# Android both ABIs (arm64: Vulkan + OpenCL + Kleidi + CPU; x86_64: Vulkan + OpenCL + CPU)
python3 tools/build.py android --abi all

# Windows x64 (Vulkan + CUDA + BLAS + CPU)
python3 tools/build.py windows --arch x64

# Windows arm64 (Vulkan + BLAS + Kleidi + CPU)
python3 tools/build.py windows --arch arm64
```

List supported combinations:

```bash
python3 tools/build.py list
```

Initialize submodules after clone:

```bash
git submodule update --init --recursive
```

## Local Linux Build With Docker Cache

Use `tools/docker_build_linux.sh` to build Linux targets in a cached Docker image.
The image keeps heavy apt dependencies (CUDA, cross toolchains, Vulkan/BLAS dev packages)
in reusable layers, so repeat builds are faster.
This Docker flow is for local development only; CI Linux jobs run on native GitHub runners.

```bash
# Linux x64 full set
./tools/docker_build_linux.sh --arch x64 --jobs 8

# Linux arm64 full set (cross-compiled in container)
./tools/docker_build_linux.sh --arch arm64 --jobs 8

# Build both Linux targets
./tools/docker_build_linux.sh --arch all --jobs 8
```

Useful flags:

- `--clean`: clean preset build directories before build
- `--rebuild-image`: force image refresh
- `--platform`: override Docker platform (default `linux/amd64`)
- `--image`: custom image tag

Outputs are written to `bin/linux/x64` and `bin/linux/arm64`.
Note: Kleidi-enabled lanes require network access to fetch upstream Kleidi sources.

Android OpenCL override env vars (optional):

- `OPENCL_INCLUDE_DIR=/path/to/opencl/headers`
- `OPENCL_LIBRARY_ANDROID_ARM64_V8A=/path/to/arm64/libOpenCL.so`
- `OPENCL_LIBRARY_ANDROID_X86_64=/path/to/x86_64/libOpenCL.so`

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/release steps
