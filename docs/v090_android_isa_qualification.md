# v0.9.0 Android CPU ISA qualification

Candidate upstream is stable v0.9.0
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`, compared with audited v0.8.0
`8642541b255bfb5dc39cd351a178be09013a55d6`. KleidiAI stays at v1.32.0 and no
dispatch input changes, so v0.9.0 grants exactly the v0.8.0 function set. Its
complete source subtree pair is:

- ggml/src: `fead7601e3b35651ee62764c76f28b50ee564160b543ff79be6fcc81837216c4`
- kai: `fd5b9ca98d07b81671eca377c258da74e72173f92f3cab0acdf6c17ab870bd9d` (v0.8.0's)

The 2026-11-05 automatic stable dispatch of v0.9.0
([run 42016701448](https://github.com/leehack/llamadart-native/actions/runs/42016701448))
failed only where this pair was unaudited: the Linux arm64 Vulkan core lane and
both Android arm64 lanes. It published nothing. This document does not claim a
published or device-qualified v0.9.0 runtime.

## Source and dispatch audit

v0.9.0 is v0.8.0 plus 150 commits. These are unchanged: the KleidiAI release
and archive MD5 in CPU CMake, `ggml-cpu/kleidiai/` (tables, probes, SME
override, thread cap, fallback mask), `arch/arm/cpu-feats.cpp`,
`arch-fallback.h`, `ggml-cpu.cpp`, the generic `repack.cpp`/`repack.h`
selection, `arch/arm/quants.c` and `ggml-backend-reg.cpp` scoring and loading.
No feature probe, selector, KleidiAI call or CMake flag changes.

The only ARM change is `arch/arm/repack.cpp`, which adds three ggml-owned
scalable paths, each behind `__ARM_FEATURE_SVE` and a runtime 256-bit vector
check:

| Function | Compile guard | Runtime guard |
| --- | --- | --- |
| `ggml_gemv_mxfp4_4x4_q8_0` | SVE | `svcntb() * 8 == 256` |
| `ggml_gemv_q8_0_4x8_q8_0` | SVE | `ggml_cpu_get_sve_cnt() == QK8_0` |
| `ggml_gemm_q6_K_8x8_q8_K` | SVE and `__ARM_FEATURE_MATMUL_INT8` | `svcntb() * 8 == 256` |

`armv8.2-a+dotprod+fp16` and every other non-SVE variant (Android
`armv8.0_1` to `armv8.6_1`, Linux arm64 baseline, `linux_armv8.2_2` and
`linux_armv8.6_1`) do not define `__ARM_FEATURE_SVE`, so these paths are not
compiled there and their NEON paths are unchanged. They compile only into the
Android SVE variants (`armv9.0_1`, `armv9.2_1`, `armv9.2_2`), which ggml scores
only on CPUs reporting SVE/SVE2, and they run only with 256-bit vectors. Android
SVE devices with 128-bit vectors take the existing NEON path. The `q6_K` GEMM's
`svmmla` is SVE I8MM, which the variant scores infer from NEON I8MM plus SVE;
v0.8.0's existing SVE I8MM paths in `arch/arm/quants.c` and `repack.cpp`
already rely on that.

The other CPU changes add no scalable code to non-SVE builds: OCP FP8 E4M3
(`GGML_TYPE_F8_E4M3`) conversion, dup and `ggml_vec_dot_f8_e4m3_f32`, whose
SIMD loop is compiled only without `__ARM_FEATURE_SVE`; `kv_rows` support in
flash attention; empty batched-copy buffer callbacks; and RISC-V guards changed
from `__riscv_v_intrinsic` to `__riscv_v`. The rest of `ggml/src` is
accelerator backends (CUDA, Metal, OpenCL, SYCL, Vulkan and others), the meta
backend, batched tensor copies in `ggml-backend.cpp`, and FP8 in `ggml.c` and
`ggml-quants.c`, none of which is in `libggml-cpu.so`. This review covers ISA
containment only, not numerical changes or accelerator backends.

## Compiled evidence

The production `build_android_arm64_cpu_variant` helper built
`android_armv8.2_2` (`armv8.2-a+dotprod+fp16`, KleidiAI on, `GGML_NATIVE=OFF`)
with NDK `28.2.13676358` on macOS arm64. Before the policy change the audit
failed with the same unaudited-pair error as the release. With the v0.9.0 pair
it passes: 265,940 instructions (v0.8.0: 261,765), with all 3,130 scalable
instructions in the same 22 exact functions. As on v0.8.0, the 20 v0.7.0
functions hold 2,542 of them and the two SVE Q8_0 kernels 201 and 387.

```bash
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source build/android-isa/_deps/kleidiai-src \
  build/android-isa/bin/libggml-cpu.so
```

`llamadart_kleidiai_dispatch_test` against v0.9.0 checked all 128 feature masks
and the 12 quantized matmuls against the scalar reference in each run:

| Host | CPU | Selected (Q4_0 / Q8_0) | Result |
| --- | --- | --- | --- |
| Apple M4 Max, native | SME2, no SVE | SME2, with and without `GGML_KLEIDIAI_SME=1` | pass |
| `ubuntu:24.04` arm64 container, GCC 13.3, `armv8-a`, QEMU 8.2.2 | `cortex-a53` | NEON baseline | pass |
| same | `max,sve=off,sme=off` | I8MM | pass |
| same | `max,sve256=on,sme=off` | SVE | pass |

Each QEMU row also passed with `GGML_KLEIDIAI_SME=1`. The `sve256` row executes
the two SVE Q8_0 kernels that v0.8.0 added. No lane had run them before.

### With patches 0005 and 0006

Re-checked 2026-11-07 on v0.9.0 with `main`'s patch series. The source pair is
unchanged because the audit fingerprints the unpatched submodule, not the
patched copies. Patch 0005 adds FMLAL/FMLAL2 (FHM) to `ggml_vec_dot_f16` only
where the compiler defines `__ARM_FEATURE_FP16_FML`: FP16 on ARMv8.4 or later.

| Module | Toolchain | Audit | FMLAL/FMLAL2 |
| --- | --- | --- | --- |
| Android `armv8.2_2` | NDK 28.2, macOS arm64 | PASS: 266,016 instructions; 3,130 scalable in the 22 functions | 0 |
| Android `armv8.6_1` | same | PASS: 269,285 instructions; 3,130 scalable in the 22 functions | 10, all in `ggml_vec_dot_f16` |
| Linux baseline, `linux_armv8.2_2`, `linux_armv8.6_1` | `ubuntu:24.04` arm64 container, GCC 13.3, `tools/check_linux_arm64_cpu_variants.sh` | PASS for all three; QEMU selects the expected module on `cortex-a53`, `cortex-a72`, `neoverse-n1`, `neoverse-v1` and `neoverse-n2`, and the finite F16 dot check passes on each | 0, 0 and 10 (`fhm=10/0`) |

`validate_linux_arm64_cpu_isa.py` allows the `fhm` family only for a variant
with FP16 on ARMv8.4 or later, so `linux_armv8.6_1` passes and an FMLAL in
`linux_armv8.2_2` would fail. `validate_android_cpu_isa.py` checks only SVE/SME
containment; it decodes with `+fp16fml` so FMLAL is not an `<unknown>` word, and
CI and the release run it on `android_armv8.2_2` only. Applying the Linux family
policy to the two Android modules above also passes, with `fhm` allowed only for
`armv8.6_1`.

## Remaining gates

- Hosted `android-arm64-isa`, `kleidiai-dispatch-emulated`,
  `windows-arm64-kleidiai` and `linux-artifact-contract (arm64, *)` v0.9.0
  rows. Windows needs no `cmake/kleidiai_windows.cmake` change because KleidiAI
  is unchanged.
- No lane runs the new ggml SVE256 repack paths. That needs an SVE variant
  under QEMU with 256-bit vectors, or Neoverse V1 hardware. Physical Android
  and SME-enabled device runs remain separate layers.
