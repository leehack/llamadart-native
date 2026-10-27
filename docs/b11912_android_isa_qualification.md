# b11912 Android CPU ISA qualification

Candidate upstream is nightly b11912
`961c376a2daa42220ec09663b9bde32c5fba1f70`, compared with audited v0.7.0
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. It uses KleidiAI v1.32.0 instead
of v1.31.0. SVE/SME stays contained: b11912 adds two SVE kernels, and both are
reachable only through a runtime-checked Q8_0 table entry. Its complete source
subtree pair is:

- ggml/src: `70f69dc5934a43a747c855eb4e50f7f33aa81a3a8c45f9d1ef5b89fa73e429f7`
- kai: `fd5b9ca98d07b81671eca377c258da74e72173f92f3cab0acdf6c17ab870bd9d`

The production submodule, release workflow and downstream pins are unchanged.
This document records local qualification. It does not claim a published or
device-qualified b11912 runtime.

## Source and dispatch audit

ARM runtime probes in `arch/arm/cpu-feats.cpp`, the ARM repack sources,
`arch-fallback.h`, backend scoring, `ggml-cpu.cpp` and `kleidiai.h` are
unchanged. CPU CMake updates the KleidiAI release and its archive MD5
(`05f1953428ffe8e8c7f585dc6d508887`). That MD5 lives in the fingerprinted
`ggml/src` tree, so it binds the KleidiAI build lists outside `kai/` too.

`ggml-feats.h` adds `has_svei8mm`, read from `HWCAP2_SVEI8MM` (bit 9) on Linux
and from `PF_ARM_SVE_I8MM_INSTRUCTIONS_AVAILABLE` (57) on Windows; Apple never
sets it. `kernels.h` adds `CPU_FEATURE_SVEI8MM = 64`. `CPU_FEATURE_SVE` is
still set only when the runtime SVE vector length is 32 bytes. Windows and
Apple report a length of 0, so they never select an SVE entry.

Table changes in `kernels.cpp`:

| Table | Change | Mask |
| --- | --- | --- |
| Q4_0 (non-Apple) | The existing SVE entry now requires SVE I8MM instead of NEON I8MM. This is stricter; the kernels are unchanged. | 7 → 69 |
| Q8_0, also used by Q6_K | New SVE entry after SME and before NEON I8MM. GEMM uses `kai_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm`; GEMV uses `kai_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot`. | 69 (SVE + SVE I8MM + DOTPROD) |
| Q4_K | Its getters move into a shared `matmul_uker_adapter`; the kernels, constants and SME2 + FP16 mask stay the same. | 48 |

The new Q8 entry packs its LHS with the existing scalar
`kai_run_lhs_quant_pack_qai8dxp_f32` and its RHS with the existing NEON
`kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon`. The GEMM N step comes from
`kai_get_sve_vector_length_u8`, which is reached only after the entry is
selected. `kleidiai.cpp` adds the SVE I8MM bit and asserts that GEMM and GEMV
share the Q8 RHS layout. Its probes, SME override, thread cap and fallback mask
subtraction are otherwise unchanged. The fallback mask is still a subset of the
runtime mask, so an SME Q8 primary can fall back to the SVE entry only on
hardware that has SVE256.

Across KleidiAI v1.31.0 → v1.32.0, no dispatched kernel or packer source
changed. The v1.31.0 tarball (MD5 `930495f3aeb16b25dc18a2bde45924f4`) hashes
to the audited v0.7.0 `kai` digest. Of 34 added files, ggml references only the
two SVE Q8 kernels. Both are built as `-march=armv8.2-a+sve` (dot) and
`+sve+i8mm` (GEMM) objects, with `.inst`-encoded assembly. The 10 modified
files are headers, docs, a `KAI_ASSUME` in `kai_div_ceil`, and a refactor of
`kai_lhs_pack_bf16p8x4_f16_neon.c`, which ggml does not use. The new `lut` and
`scale_nk` argument fields are zero-initialised by ggml's adapters.

Other CPU deltas add no reachable scalable path. SVE GELU-quick is guarded by
`__ARM_FEATURE_SVE`, which the armv8.2 variant does not define. The rest are
NEON GELU-quick, s390x SIMD, `MUL_MAT_ID` handling of id -1, `TOP_K_MASK`
and FP16 conversion-macro checks. This review covers ISA containment only. It
does not cover numerical changes or accelerator backends.

## Compiled evidence

The production `build_android_arm64_cpu_variant` helper built
`android_armv8.2_2` (`armv8.2-a+dotprod+fp16`, KleidiAI on, `GGML_NATIVE=OFF`)
with NDK `28.2.13676358` on macOS arm64. The unmodified CLI rejected the exact
pair above, reproducing the candidate lane failure. With the reviewed policy it
passed: 261,695 instructions, with all 3,130 scalable instructions in 22 exact
functions. That is the 20 v0.7.0 functions, still with 2,542 instructions, plus
`kai_kernel_matmul_clamp_f32_qai8dxp1x8_qsi8cxp8x8_1x8_sve_dot` (201) and
`kai_kernel_matmul_clamp_f32_qai8dxp4x8sf32_qsi8cxp4vsx8sf32bf32_16x4vs_sve_i8mm`
(387). The validator grants these two only to the b11912 pair. Legacy and v0.7.0
pairs keep their own sets, and cross-version pairs are rejected.

```bash
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source build/android-isa-b11912/_deps/kleidiai-src \
  build/android-isa-b11912/bin/libggml-cpu.so
```

The compiled dispatch test detects `CPU_FEATURE_SVEI8MM` with a declaration
probe. Against b11912 it checks all 128 masks, expecting mask 69 for the SVE
Q4_0 and Q8_0/Q6_K entries; older headers keep 64 masks and mask 7.
A mutant that drops the Q8 mask-69 expectation fails the selector check.

On an Apple M4 Max with `GGML_KLEIDIAI_SME=0`, every selector check and all 12
matmuls passed. Q4_0 ran through DOTPROD, Q8_0 and Q6_K through I8MM, and Q4_K
fell back to the baseline buffer. With SME enabled, they passed through SME2.
The CPU reports no SVE, so the new SVE kernels did not execute. The aarch64 GCC
plus QEMU lane could not run locally: this host has no `qemu-aarch64`, no
`aarch64-linux-gnu` toolchain and no running Docker daemon.

## Remaining gates

- Hosted `android-arm64-isa`, `kleidiai-dispatch-emulated` (`cortex-a53` and
  `max,sve=off,sme=off`) and `windows-arm64-kleidiai` b11912 rows.
  Windows compiles the two new `.S` files through upstream's MSVC lists and the
  existing ClangCL MARMASM settings; `cmake/kleidiai_windows.cmake` needs no
  change.
- No lane executes the new SVE Q8 kernels. That requires SVE256 with SVE I8MM,
  for example QEMU with `sve-default-vector-length=32,sme=off` or Neoverse V1
  hardware. Android SVE devices with 128-bit vectors never select them.
  Physical Android and SME-enabled execution remain separate layers.
