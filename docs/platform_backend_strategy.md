# Platform Backend Strategy

## Worthy Backend Sets (Shipped Per Target)

| Platform | Built backends |
|---|---|
| Android arm64 | Vulkan + OpenCL + CPU variants (Kleidi-enabled where safe) |
| Android x64 | Vulkan + OpenCL + CPU |
| iOS | Metal + CPU |
| macOS | Metal + CPU |
| Linux x64 | Vulkan + CUDA + BLAS + CPU (HIP/ROCm built in a separate release job) |
| Linux arm64 | Vulkan + BLAS + Kleidi + CPU |
| Windows x64 | Vulkan + CUDA + BLAS + CPU |
| Windows arm64 | Vulkan + BLAS + Kleidi + CPU |

## Build Model

- Release CI builds non-Apple backend lanes separately and merges them into
  per-platform/arch bundles. Local `full` builds use one preset for the listed
  non-HIP backends.
- Apple (iOS/macOS): consolidate Metal+CPU into a single `libllamadart`.
- Apple defaults keep BLAS and Kleidi disabled for a simpler compatibility path.
- Kleidi is enabled on Linux arm64 and Windows arm64 in this pipeline.
- Android arm64 keeps Kleidi on by building each CPU variant in its own
  isolated configuration so higher-tier ISA flags do not leak into lower-tier
  variant binaries.
- Non-Apple: keep backends as separate dynamic libraries (`GGML_BACKEND_DL=ON`).

## Runtime Packaging Model

- Apple: ship only `libllamadart` for each target.
- Non-Apple required core libs: `llamadart`, `llama`, `llama-common`, `ggml`, `ggml-base` (and `mtmd` when present).
- Non-Apple optional backend libs: `ggml-<backend>` modules (for example `ggml-vulkan`, `ggml-opencl`, `ggml-cuda`).
- App integrators decide which backend modules to ship and load at runtime.

## Constraints

- CUDA lanes require `nvcc` availability.
- HIP/ROCm lanes require `hipcc`, `rocblas-dev`, and `hipblas-dev` (Linux x64 only, built in a separate release job).
- Android Vulkan lanes require NDK-provided `libvulkan.so`.
- Vulkan lanes use vendored `third_party/SPIRV-Headers` for SPIR-V registry headers required by upstream `llama.cpp`.
- Android OpenCL lanes require `CL/cl.h` and `libOpenCL.so` from one of:
  - env overrides (`OPENCL_INCLUDE_DIR`, `OPENCL_LIBRARY_ANDROID_<ABI>`)
  - `third_party/opencl-stubs/`
  - auto-built OpenCL ICD loader from `third_party/OpenCL-ICD-Loader` + `third_party/OpenCL-Headers`
- Linux arm64 builds on x64 runners require `aarch64-linux-gnu-gcc/g++`, `libopenblas-dev:arm64`, and `libvulkan-dev:arm64`.
- Windows MSVC builds disable IPO/LTCG for `llama-common` and `mtmd` by default.
  Current MSVC `link.exe` can access-violate when linking the large
  `llama-common` utility DLL with `/LTCG`. CMake's automatic Windows export
  scanner also cannot generate `mtmd.dll` exports from LTCG object files. Use
  `LLAMADART_MSVC_LLAMA_COMMON_IPO=ON` or
  `LLAMADART_MSVC_MTMD_IPO=ON` only when retesting a newer compiler or upstream
  change.

## Dependency Management

- Native dependencies are pinned as git submodules under `third_party/`.
