# v1.1.0 ARM source and ISA audit

The fixed candidate is llama.cpp `171527522610996f5f0acf12334e185156ca8f44`
(resolved stable v1.1.0), compared with v1.0.0
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28`. KleidiAI remains v1.32.0 (`kai`
digest unchanged). The reviewed complete unpatched source pair is:

- ggml/src: `4556ff74f100cb77202657b26bcc77c715c7321b1a6a0f9f9ed6243f7f2eebd0`
- kai: `fd5b9ca98d07b81671eca377c258da74e72173f92f3cab0acdf6c17ab870bd9d`

## Source review

Byte-identical to v1.0.0: the `ggml-cpu/kleidiai/` sources (including
selectors, kernel tables and masks), `ggml-cpu/arch/arm/` except `quants.c` and
`repack.cpp`, `ggml-cpu-impl.h` and `kleidiai/`'s caller code. The CPU CMake
changes are `OHOS` in a platform list and `-w` on the `kleidiai` target; neither
changes ISA flags.

Changed ARM-reachable code, all reviewed:

- `arch/arm/quants.c` adds SVE `Q5_0 x Q8_0` and `Q5_1 x Q8_1` dot kernels and a
  `load_as_svbool` helper. Every SVE intrinsic is under
  `#if defined(__ARM_FEATURE_SVE)`, with the NEON path kept in `#elif`. The
  release non-SVE variant does not define that flag, so these kernels exist only
  in modules compiled with SVE flags and are reached only through the existing
  CPU-feature variant scoring. They branch on `ggml_cpu_get_sve_cnt()` for
  128/256/512-bit vectors.
- `arch/arm/repack.cpp` adds NEON repack/GEMM variants under
  `!__ARM_FEATURE_DOTPROD` compile guards, which `repack.cpp` selects at run
  time when `ggml_cpu_has_neon() && !ggml_cpu_has_dotprod()` on non-Apple
  AArch64. NEON only; no scalable register, and the dotprod release variants do
  not compile them.
- `ggml-cpu.c/.cpp`, `ops.cpp/.h`, `vec.h` and `repack.cpp` add operators,
  BF16 handling, GDN replay and the MUL_MAT blocking change of
  [#30846](https://github.com/ggml-org/llama.cpp/pull/30846). They add no ARM
  feature probe, no scalable intrinsic and no Kleidi caller.
- `llamafile/sgemm.cpp` adds F16 x F32 and s390 VXE work; it has no new ARM
  branch, and the ARM F16 x F16 branch that carry 0006 patches is unchanged.
- The `wasm/`, `x86/` and `spacemit/` files are outside the ARM targets.

The rest of the fingerprinted `ggml/src` tree changed in accelerator backends
(CUDA, Metal, Vulkan, SYCL, Hexagon, OpenCL and others) and does not change the
dispatch of the standalone ARM CPU module.

## Compiled controls

2026-11-26, Apple M4 Max / macOS 27.2.1, AppleClang 21.0.0.
`llamadart_kleidiai_dispatch_test` links the candidate CPU library, with the
production configuration `GGML_CPU_KLEIDIAI=ON` and
`LLAMADART_BUILD_KLEIDIAI_TESTS=ON`. It passes all **128 feature masks**,
positive and negative tensor paths and **12** Q4_0/Q8_0/Q4_K/Q6_K
matrix/scalar-reference comparisons (13 PASS lines). The same binary passes with
`GGML_KLEIDIAI_SME=0` for the non-SME control. Hardware reports SVE absent and
SME/SME2 present; this is not execution on hardware lacking SME, nor of an SVE
kernel, and Q4_K takes the baseline fallback.

The production `build_android_arm64_cpu_variant` helper built
`android_armv8.2_2` (`armv8.2-a+dotprod+fp16`, Kleidi on, native off, llamafile
off, Android API 28, NDK **28.2.13676358**) on macOS arm64; KleidiAI v1.32.0 was
fetched by upstream's FetchContent and its `kai` digest equals the audited
pair. The module SHA-256 is
`23d0ecd0bc709eefde8c325a737ff1a20e26196882da2e6deab7501409cab038`.

The unchanged validator first rejected that module with the unaudited
source-pair error. After the review above, adding only this exact pair passes:
**270778 instructions**, **3130 SVE/SME instructions inside the same 22 exact
audited functions** as v1.0.0 (v0.8.0's allowed set). No function-range or
instruction-family exception is added. All earlier pairs and the unknown,
cross-combined, added-caller, changed-kernel and escaped-instruction negative
controls remain in `tests/test_android_cpu_isa.py`.

```sh
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source <build>/_deps/kleidiai-src \
  <build>/bin/libggml-cpu.so
```

This admits source identity and establishes bounded compiled containment of
the non-SVE module. It does not qualify Android execution, a published archive
or every platform.

## Remaining gates

- The hosted `validate_wrapper.yml` lanes must pass at the final head: Android
  artifact/disassembly, Windows ARM64 ClangCL, Linux variants and the QEMU
  non-SVE selector/compute run.
- The new SVE `Q5_0`/`Q5_1` kernels have never run on SVE hardware by these
  gates. Linux arm64 SVE variants and the Android SVE modules are compiled with
  `__ARM_FEATURE_SVE` and need the separate ggml SVE256 controls and a
  128-bit-vector SVE device (Pixel 9 Pro class) before the pair is called
  qualified for them ([#145](https://github.com/leehack/llamadart-native/issues/145)).
- Physical provider performance and original numerical/model/device triggers
  remain in [v110_qualification.md](v110_qualification.md).
