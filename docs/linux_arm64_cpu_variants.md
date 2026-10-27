# Linux arm64 CPU variants

Linux arm64 bundles ship the ARMv8.0 CPU backend plus higher-ISA variants that
ggml selects at run time. Before this, the only `libggml-cpu.so` was built for
ARMv8.0 and had no FP16 vector arithmetic, so F16 matmuls on Neoverse-N1 ran
about 37x slower than on Apple CPUs (#126, leehack/llamadart#675).

| File | `-march` | Scored features | Selected on, for example |
|---|---|---|---|
| `libggml-cpu.so` | compiler default (`armv8-a`) | none, unscored | Cortex-A53/A72 (Raspberry Pi 4), or when no variant scores |
| `libggml-cpu-linux_armv8.2_2.so` | `armv8.2-a+dotprod+fp16` | DOTPROD, FP16 | Neoverse-N1 (Graviton2, Ampere Altra, GCE T2A), Cortex-A76 (Raspberry Pi 5) |
| `libggml-cpu-linux_armv8.6_1.so` | `armv8.6-a+dotprod+fp16+i8mm` | DOTPROD, FP16, I8MM | Neoverse-V1/N2/V2 (Graviton3/4, Axion, Grace) |

The feature sets match Android's `android_armv8.2_2` and `android_armv8.6_1`.
`tools/build.py` builds each variant in its own CMake configure, as on Android,
so the shared KleidiAI target never inherits another variant's `-march`. Only
the lanes that ship `libggml-cpu.so` (`full`, `cpu`, `vulkan`) build them. Each
variant adds about 1.2 MB uncompressed.

SVE variants are not shipped. ggml's SVE paths and the KleidiAI kernels they
reach need their own audit and measurement on 128-bit and 256-bit SVE parts.
Upstream also compiles out the llamafile SGEMM in I8MM builds, so the
`linux_armv8.6_1` F16 matmul runs on ggml's own FP16 vector dot product and its
F32 GEMM loses the llamafile path.

## Compiler-emitted instructions beyond the scored features

The scored features are not the only instructions a `-march` base version
enables. The compiler also uses these where they apply:

- `linux_armv8.2_2` (`armv8.2-a`) uses inline LSE atomics, which ARMv8.1
  made mandatory. Every CPU that scores this variant has them.
- `linux_armv8.6_1` (`armv8.6-a`) additionally uses 31 `ldapr` (LRCPC, ARMv8.3)
  and 3 `stlur` (LRCPC2, ARMv8.4) instructions, plus the same inline LSE.
  Neither is a scored feature; ARMv8.6 guarantees both, and the I8MM that
  scores this variant is an ARMv8.6 feature.
- `libggml-cpu.so` (ARMv8.0) contains LSE only inside libgcc's outline-atomic
  helpers, each of which tests `__aarch64_have_lse_atomics` before reaching the
  LSE instruction. It contains no LRCPC, LRCPC2 or CRC.

## Measured performance

Neoverse-N1 (GCE `t2a-standard-8`, 2026-10-27), where auto selection picks
`libggml-cpu-linux_armv8.2_2.so`
([PR 127 comment](https://github.com/leehack/llamadart-native/pull/127#issuecomment-6440106818)):

| | v0.7.0-2 baseline | `linux_armv8.2_2` |
|---|---:|---:|
| Qwen2.5-0.5B F16 pp512 | 198.5 t/s | 328.9 t/s (1.66x) |
| Qwen2.5-0.5B F16 tg128 | 35.7 t/s | 70.2 t/s (1.97x) |
| Qwen2.5-0.5B Q8_0 pp512 / tg128 | 1143 / 126.3 t/s | 1144 / 126.6 t/s (unchanged) |
| Qwen3-TTS in-flight cancel, worst of 9 (500 ms budget) | 697 ms, fails | 180 ms, passes |
| Qwen3-TTS longest code2wav chunk | about 1300 ms | about 203 ms |
| Qwen3-TTS cancelled synthesis, total | about 11.3 s | about 4.8 s |

The TTS rows force the variant by replacing `libggml-cpu.so` with it. With this
layout, llamadart's explicit `--backend cpu` mode still loads the baseline
(worst cancel 730 ms) until llamadart ranks Linux arm64 variants first.

`linux_armv8.6_1` is not measured. On Neoverse-V1/N2 its F32 GEMM may run
slower than the baseline's, because I8MM builds lose the llamafile SGEMM; this
is unmeasured.

## Selection and the consumer contract

The baseline keeps the unsuffixed name, so the change is additive and the
release keeps `hook_contract_version` unchanged:

- `ggml_backend_load_all_from_path()` (llamadart's `GpuBackend.auto`) opens
  every `libggml-cpu-*.so`, calls its `ggml_backend_score` and loads the
  highest non-zero score. When none scores, it loads `libggml-cpu.so`.
- `ggml_backend_load("libggml-cpu.so")` loads the baseline, as before.
  llamadart's explicit modes (`GpuBackend.cpu`, and the CPU module the
  `vulkan`/`blas` modes load) try `libggml-cpu.so` first and so stay on the
  baseline until llamadart ranks Linux arm64 variants the way it ranks
  Android's.
- llamadart's hook infers backend `cpu` from every `libggml-cpu*` file and
  bundles them all for `linux-arm64`; it filters CPU variants only for
  `android-arm64`.

A process that loads models in both auto and explicit CPU mode registers two
CPU backends, the selected variant and the baseline. ggml and llama.cpp use
the first CPU device registered.

## ISA containment

`tools/check_linux_arm64_cpu_variants.sh <bundle-dir>` runs both checks below.
`linux-artifact-contract (arm64, *)` in `validate_wrapper.yml` runs it on every
qualified upstream, and the release's Linux arm64 core lane runs it before
upload.

`tools/validate_linux_arm64_cpu_isa.py` audits each module. Outside sized
`kai_*` functions, FP16 arithmetic, FP16FML, DOTPROD, I8MM, BF16, RDM,
LSE, CRC, LRCPC and LRCPC2 instructions may appear only when the variant's
`-march` enables them. `linux_armv8.2_2` may contain LSE, and `linux_armv8.6_1`
LSE, LRCPC and LRCPC2. The baseline may contain none of them, except LSE inside
libgcc's outline-atomic helpers, which the validator recognizes by their
`adrp x16`, `ldrb w16`, `cbz w16` guard. KleidiAI kernels may contain them
because KleidiAI picks kernels from the runtime HWCAP probe, not from compile
flags. SVE and SME stay in the exact audited functions and pinned source pairs
of `tools/validate_android_cpu_isa.py`. Each variant must also contain its
declared families outside KleidiAI and export `ggml_backend_score`.

The script then runs `tools/linux_cpu_variant_smoke.c` under QEMU, which raises
SIGILL for any instruction the emulated CPU lacks. Selection loads every
variant and calls its score function, so a passing run also shows that the
higher variants load safely on lower CPUs.

| `qemu-aarch64 -cpu` | Selected | F16 x F32 matmul |
|---|---|---|
| `cortex-a53`, `cortex-a72` | `libggml-cpu.so` | passes |
| `neoverse-n1` | `libggml-cpu-linux_armv8.2_2.so` | passes |
| `neoverse-v1`, `neoverse-n2` | `libggml-cpu-linux_armv8.6_1.so` | passes |

QEMU shows CPU compatibility, not speed. The speedup needs a run on real
Neoverse-N1 hardware.
