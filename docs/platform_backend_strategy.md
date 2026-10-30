# Platform Backend Strategy

## Worthy Backend Sets (Shipped Per Target)

| Platform | Built backends |
|---|---|
| Android arm64 | Vulkan + OpenCL + CPU variants (Kleidi-enabled where safe) |
| Android x64 | Vulkan + OpenCL + CPU |
| iOS | Metal + CPU |
| macOS | Metal + CPU |
| Linux x64 | Vulkan + CUDA + BLAS + CPU (HIP/ROCm built in a separate release job) |
| Linux arm64 | Vulkan + BLAS + Kleidi + CPU variants |
| Windows x64 | Vulkan + CUDA + BLAS + CPU |
| Windows arm64 | Vulkan + BLAS + Kleidi + CPU |

## Build Model

- Release CI builds non-Apple backend lanes separately and merges them into
  per-platform/arch bundles. Local `full` builds use one preset for the listed
  non-HIP backends.
- Apple (iOS/macOS): consolidate Metal+CPU into a single `libllamadart`.
- Apple defaults keep BLAS and Kleidi disabled for a simpler compatibility path.
- Kleidi is enabled on Linux arm64 and Windows arm64 in this pipeline.
- Linux arm64 adds DOTPROD+FP16 and I8MM CPU variants beside the ARMv8.0
  `libggml-cpu.so`, each built in its own configuration and selected by
  `ggml_backend_score`. See `docs/linux_arm64_cpu_variants.md`.
- Android arm64 keeps Kleidi on by building each CPU variant in its own
  isolated configuration so higher-tier ISA flags do not leak into lower-tier
  ggml code. With llama.cpp v0.4.0, standalone KleidiAI also contains kernels
  selected by runtime CPU features. The Android ISA audit permits scalable
  instructions only inside exact reviewed ELF function ranges and binds that
  exception to the complete audited ggml/Kleidi source fingerprints. Unknown
  ranges or changed source fail closed; legacy artifacts without scalable code
  continue through the strict path. Never refresh fingerprints without reviewing
  feature detection, kernel tables and callers, and passing compiled dispatch
  tests and non-SVE compute qualification.
- Windows ARM64 retains ClangCL and all Kleidi kernels. Source-local Visual
  Studio metadata selects the native ARMASM preprocessing dialect and suppresses
  incompatible line markers. The owning CMake integration also restores the
  exact ClangCL-compatible C kernels referenced by ggml but omitted from
  KleidiAI's MSVC source list, preserving their upstream per-source ISA flags
  without raising the baseline ISA for other code.
- Non-Apple: keep backends as separate dynamic libraries (`GGML_BACKEND_DL=ON`).

## ARM64 upgrade qualification

`Validate Wrapper` checks pinned and exact post-v0.4.0 Windows ARM64 builds,
the actual Android ARMv8.2 artifact, and compiled Kleidi selectors plus quantized
matrix computation under non-SVE QEMU profiles. QEMU is deterministic CPU
compatibility evidence, not physical-device or GPU-performance evidence.
Wrapper contract assertions stay active in Release builds; Windows CTest resolves
both wrapper and upstream DLL directories and bounds each test to 120 seconds.
Use `-DLLAMADART_BUILD_KLEIDIAI_TESTS=ON` with standalone-Kleidi upstream to
build `llamadart_kleidiai_dispatch_test`; it links the actual CPU module.
The Android artifact check is `tools/validate_android_cpu_isa.py --help`.
The Linux arm64 variant check is `tools/check_linux_arm64_cpu_variants.sh`.
Version-specific source and function-range evidence for v0.7.0 lives in
`docs/v070_android_isa_qualification.md`; its candidate lanes retain the older
qualified releases as separate regression inputs.

## Runtime Packaging Model

- Apple: ship only `libllamadart` for each target.
- Non-Apple required core libs: `llamadart`, `llama`, `llama-common`, `ggml`, `ggml-base` (and `mtmd` when present).
- Non-Apple optional backend libs: `ggml-<backend>` modules (for example `ggml-vulkan`, `ggml-opencl`, `ggml-cuda`).
- App integrators decide which backend modules to ship and load at runtime.

## Constraints

- CUDA lanes require `nvcc` availability.
- HIP/ROCm lanes require `hipcc`, `rocblas-dev`, and `hipblas-dev` (Linux x64 only, built in a separate release job).
- Android Vulkan lanes require NDK-provided `libvulkan.so`.
- Android Vulkan shaders need a glslc with `GL_KHR_cooperative_matrix` from
  llama.cpp v0.5.0 on: upstream compiles the `fa_decode` shaders as coopmat
  unconditionally (ggml-org/llama.cpp#29373), and NDK glslc (shaderc v2022.3
  through r29) lacks the extension. Set `ANDROID_VULKAN_GLSLC` to a host glslc;
  release CI uses the Ubuntu 24.04 `glslc` package (shaderc 2023.8), like the
  Linux Vulkan lanes, on a pinned `ubuntu-24.04` runner so the shader feature set
  cannot drift with `ubuntu-latest`. This applies to every release built with
  the override, including rebuilds of the current pin.
  Upstream enables shader features from glslc, so this also compiles the KHR
  coopmat `mul_mm`/flash-attention variants, used at runtime only on devices
  reporting `VK_KHR_cooperative_matrix` (`GGML_VK_DISABLE_COOPMAT=1` opts out).
- Vulkan lanes use vendored `third_party/SPIRV-Headers` for SPIR-V registry headers required by upstream `llama.cpp`.
- Android OpenCL lanes require `CL/cl.h` and `libOpenCL.so` from one of:
  - env overrides (`OPENCL_INCLUDE_DIR`, `OPENCL_LIBRARY_ANDROID_<ABI>`)
  - `third_party/opencl-stubs/`
  - auto-built OpenCL ICD loader from `third_party/OpenCL-ICD-Loader` + `third_party/OpenCL-Headers`
- Linux arm64 builds on x64 runners require `aarch64-linux-gnu-gcc/g++`, `libopenblas-dev:arm64`, and `libvulkan-dev:arm64`.
- Release Linux builds (`build-linux`) and the PR lanes that stand in for them
  (`linux-artifact-contract`, `kleidiai-dispatch-emulated`, and
  `build-linux` in `validate_linux_release_rows.yml`) run on a pinned
  `ubuntu-24.04` runner (GCC 13, glibc 2.39). On the Ubuntu 26.04 image that
  `ubuntu-latest` moved to in October 2026 (#132):
  - CUDA 12.8 `nvcc` rejects the host GCC 15.
  - GCC 14 and 15 emit SVE (`and z29.s, z29.s, #0x3f`) in the local
    `pack_metadata` routine of KleidiAI's
    `kai_matmul_pack_rhs_nxk_qai4c32p16vsx4s1s0sf16_qai4c32k256sf16s32s0_sme.c`,
    which KleidiAI compiles with `-march=armv8.2-a+sve+sve2`. The baseline ISA
    audit reads `.dynsym` and cannot attribute a local routine, so it fails.
    GCC 13 emits no SVE in that file. ggml reaches the routine only through
    `q4_k_rhs_pack`, whose one table entry requires `SME2 | FP16` (v0.7.0 and
    v0.8.0; KleidiAI v1.31.0 and v1.32.0 have the same file). That gate does
    not make the SVE safe: the routine runs outside streaming mode, and SME2
    does not imply SVE there. On an Apple M4 Max (SME2, no FEAT_SVE) this
    instruction raises SIGILL outside streaming mode and runs inside it.
  - The arm64 `libggml-cpu.so` needs `sqrtf@GLIBC_2.43`, so it no longer loads
    on Ubuntu 24.04 or Debian 13.
  - The arm64 multiarch dependency install stops because apt would downgrade
    `make` and `pkg-config`.

  Unpin only with a CUDA toolkit that supports the image's GCC, a build whose
  `pack_metadata` has no SVE outside streaming mode (or a kernel gate that also
  requires SVE), a Linux glibc floor raise announced to `llamadart` consumers,
  and an arm64 multiarch dependency install that no longer downgrades
  `libpkgconf7 make pkg-config pkgconf`. Admitting that SVE to the ISA audit by
  an exact range is not enough on its own.

  The floor is `GLIBC_2.38`, the highest version the 24.04 build imports.
  `tools/check_glibc_floor.py` fails a Linux `.so` above it in `build-linux`,
  `linux-artifact-contract` and `validate_linux_release_rows.yml`.
  `tools/validate_linux_artifact.py` applies the same floor to the merged
  bundles at release, which is what covers `libggml-hip.so` from the ROCm
  container. Raise `GLIBC_FLOOR` in the same change that announces the new
  floor.

  `validate_linux_release_rows.yml` runs the release's Linux job bodies without
  their upload, weekly and on pull requests that change their inputs, plus
  CUDA and HIP at the v0.8.0 candidate. `tests/test_linux_release_rows.py` fails
  when it stops matching `native_release.yml`.
  `tests/test_runner_pins.py` fails when any of these jobs, or the Android
  `build-android` and `android-vulkan-shaders` jobs, leaves `ubuntu-24.04`.
- Windows ARM64 uses the `Visual Studio 18 2026` generator (CMake >= 4.2) with
  the ClangCL toolset. CI pins the `windows-11-vs2026-arm` runner because the
  `windows-11-arm` label moved to that image in September 2026
  (actions/runner-images#14602), and that image has no VS 2022 instance.
  `tools/build.py windows --arch arm64` therefore needs VS 2026. With only
  VS 2022 installed, configure directly with
  `cmake --preset windows-arm64-full -G "Visual Studio 17 2022"` in a fresh
  build directory.
- Windows ARM64 Vulkan links LunarG's native ARM64 SDK (`warm` platform), whose
  `vulkan-1.lib` is ARM64. The x64 SDK's import library is x64-only, which
  VS 2026's ClangCL linker (`lld-link`) rejects. `.github/actions/install-vulkan-sdk-windows`
  pins both installers by SHA-256 and checks the import library's machine type.
- Windows MSVC builds disable IPO/LTCG for `llama-common` and `mtmd` by default.
  Current MSVC `link.exe` can access-violate when linking the large
  `llama-common` utility DLL with `/LTCG`. CMake's automatic Windows export
  scanner also cannot generate `mtmd.dll` exports from LTCG object files. Use
  `LLAMADART_MSVC_LLAMA_COMMON_IPO=ON` or
  `LLAMADART_MSVC_MTMD_IPO=ON` only when retesting a newer compiler or upstream
  change.

## Dependency Management

- Native dependencies are pinned as git submodules under `third_party/`.
