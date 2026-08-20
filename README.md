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
  - Pins every build to one resolved `llama.cpp` commit and publishes a source
    tag whose submodule entry identifies that exact commit.
- `Auto Trigger Native Release` (`.github/workflows/auto_native_release.yml`)
  - Daily schedule plus manual dispatch.
  - Resolves latest upstream `ggml-org/llama.cpp` release tag.
  - Dispatches `Native Build & Release` only when this repo does not already have that tag and no native release run is in flight.

## Native Version Management

The published tag is the native version contract consumed by downstream package
hooks and Swift Package manifests. `llama_cpp_tag` selects the upstream
`llama.cpp` ref to build. `native_release_tag` selects the
`llamadart-native` release tag and archive suffix; when blank, it defaults to
the resolved `llama_cpp_tag`.

When changing the upstream `llama.cpp` version:

1. Run `Native Build & Release` for the selected `llama_cpp_tag`, or let
   `Auto Trigger Native Release` dispatch it for the latest upstream tag.
2. Verify the release contains per-platform native archives,
   `llamadart-native-apple-xcframework-<tag>.zip`, `assets.json`, and
   `SHA256SUMS`.
3. Update downstream `llamadart` pins, SPM URLs, and SPM checksums together so
   native-assets and SPM consumers use the same wrapper/runtime build.

When rebuilding the wrapper without changing the upstream `llama.cpp` ref,
dispatch `Native Build & Release` with the same `llama_cpp_tag` and a new
`native_release_tag`, for example `b9873-llamadart.1`. Do not republish
different source under an existing raw upstream tag; downstream caches are
tag-keyed and the GitHub source tag should identify the native wrapper commit
that produced the assets. Publish mode fails if the selected
`native_release_tag` already exists as a tag or GitHub Release.

`assets.json` records both the requested `llama_cpp_tag` and its resolved
`llama_cpp_commit`, plus the `native_commit` targeted by the published source
tag. Consumers can therefore verify the built source independently of a moving
ref or release label.

## Backend Policy (Worthy Sets)

Each target builds all worthy backends together in one build:

- Android: arm64 = Vulkan + OpenCL + CPU variants (Kleidi-enabled where safe); x86_64 = Vulkan + OpenCL + CPU
- iOS/macOS: Metal + CPU (consolidated into `libllamadart`, BLAS/Kleidi disabled)
- Linux x64: Vulkan + CUDA + BLAS + ZenDNN + CPU
- Linux arm64: Vulkan + BLAS + Kleidi + CPU
- Windows x64: Vulkan + CUDA + BLAS + CPU
- Windows arm64: Vulkan + BLAS + Kleidi + CPU

Non-Apple targets use `GGML_BACKEND_DL=ON`, so backend libs are optional at package/runtime level.

## Runtime Packaging Model

Release assets contain:

- Apple: consolidated `libllamadart` per target.
- Apple SPM: `llamadart-native-apple-xcframework-<tag>.zip`, a
  `llamadart_native.xcframework` built from the same Apple slices and wrapper
  code as the native-assets tarballs.
- Non-Apple core libs: `llamadart`, `llama`, `llama-common`, `ggml`, `ggml-base` (and `mtmd` where produced)
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
- `third_party/Vulkan-Headers`: Vulkan API headers submodule for Android Vulkan builds.
- `third_party/SPIRV-Headers`: SPIR-V registry headers required by the `llama.cpp` Vulkan backend.
- `third_party/OpenCL-Headers`: OpenCL headers submodule (Android OpenCL builds).
- `third_party/OpenCL-ICD-Loader`: OpenCL loader submodule used to produce Android `libOpenCL.so` when NDK does not provide one.
- `third_party/opencl-stubs`: optional local fallback location for OpenCL headers/stubs.
- `tools/build.py`: cross-platform build entrypoint.
- `tools/validate_exports.py`: verifies required wrapper C exports, including
  speculative-decoding and TTS symbols, in release artifacts.
- `tools/package_apple_xcframework.py`: packages Apple `libllamadart` slices as
  an SPM-compatible XCFramework zip.
- `scripts/generate_assets_manifest.sh`: builds `assets.json` + checksums.
- `scripts/verify_release_provenance.py`: verifies exact-source checkout, source
  tag, and manifest provenance contracts.
- `docs/platform_backend_strategy.md`: platform/backend matrix.

## Local Build (Preferred)

Builds are primarily driven by root `CMakePresets.json` via `tools/build.py`.
Android arm64 CPU variants use isolated CMake build directories so per-variant
ISA flags remain correct while packaging the full variant matrix. The raw
`android-arm64-v8a-full` preset now represents the primary arm64 build, while
`tools/build.py` assembles the additional CPU variant outputs.

Examples:

```bash
# macOS arm64 (Metal + CPU)
python3 tools/build.py apple --target macos-arm64

# Linux x64 (Vulkan + CUDA + BLAS + ZenDNN + CPU)
python3 tools/build.py linux --arch x64

# Android both ABIs (arm64: Vulkan + OpenCL + CPU variants; x86_64: Vulkan + OpenCL + CPU)
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

## Experimental TTS Wrapper

`libllamadart` exposes a versioned, opaque C symbol contract around llama.cpp's
experimental audio-generation helpers. The wrapper currently recognizes
Qwen3-TTS projectors, exposes model capability metadata, and
provides task start/step/cancel/reset plus caller-buffered float32 mono PCM
reads after synthesis completes. The caller owns the `llama_context` and
`mtmd_context`, must create the llama context with embeddings enabled, and must
give the TTS task exclusive access to both contexts until the task reaches a
terminal state or is reset.

The upstream API does not currently expose completed PCM incrementally, so the
wrapper's step API is cancellable between prompt batches and generation frames
but is not a real-time audio stream. Public Dart support should remain
experimental and capability-gated until artifact and platform validation is
complete.

The local smoke target is opt-in because it requires large model artifacts:

```bash
cmake -S . -B build/tts-smoke -G Ninja \
  -DGGML_CCACHE=OFF \
  -DGGML_METAL=OFF \
  -DLLAMADART_BUILD_TESTS=ON \
  -DLLAMADART_BUILD_TTS_SMOKE=ON
cmake --build build/tts-smoke --target \
  llamadart_speculative_api_test llamadart_tts_api_test llamadart_tts_smoke
ctest --test-dir build/tts-smoke --output-on-failure
build/tts-smoke/llamadart_tts_smoke \
  /path/to/Qwen3-TTS-model.gguf \
  /path/to/mmproj-Qwen3-TTS.gguf \
  /tmp/tts-smoke.wav \
  "Hello from Llama Dart." en \
  --gpu
```

The smoke checks capability metadata, cancellation/reset, two consecutive
syntheses, 24 kHz mono PCM metadata, finite/non-silent output, and WAV writing.
Omit `--gpu` for a CPU-only run. An optional speaker-reference audio path may
appear before the final `--gpu` flag.

## Windows Build Notes

MSVC release builds keep interprocedural optimization enabled by default, but
`llama-common` and `mtmd` are excluded from IPO/LTCG. Upstream `llama-common` is
a large utility DLL, and current MSVC `link.exe` can access-violate while
linking it with `/LTCG`. Upstream `mtmd` enables CMake's automatic Windows
symbol export; CMake cannot generate the export definition from LTCG object
files. These target-scoped overrides keep Windows release artifacts
reproducible without changing the runtime packaging model.

To retest MSVC IPO after a compiler or upstream change, configure with:

```bash
cmake --preset windows-x64-full \
  -DLLAMADART_MSVC_LLAMA_COMMON_IPO=ON \
  -DLLAMADART_MSVC_MTMD_IPO=ON
```

## Local Linux Build With Docker Cache

Use `tools/docker_build_linux.sh` to build Linux targets in a cached Docker
image. The image is based on NVIDIA CUDA 12.8.1 and keeps heavy dependencies
(CUDA, cross toolchains, Vulkan/BLAS dev packages) in reusable layers, so repeat
builds are faster.
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
Android arm64 CPU variants are built in isolated configurations so Kleidi can
stay enabled without leaking newer ISA flags into lower-tier variant binaries.

Android OpenCL override env vars (optional):

- `OPENCL_INCLUDE_DIR=/path/to/opencl/headers`
- `OPENCL_LIBRARY_ANDROID_ARM64_V8A=/path/to/arm64/libOpenCL.so`
- `OPENCL_LIBRARY_ANDROID_X86_64=/path/to/x86_64/libOpenCL.so`

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/release steps
