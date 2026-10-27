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
`linux_armv8.6_1` F16 matmul runs on ggml's own FP16 vector dot product.

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
`kai_*` functions, FP16 arithmetic, FP16FML, DOTPROD, I8MM, BF16, RDM and
LRCPC2 instructions may appear only when the variant's `-march` enables them,
so the baseline may contain none of them. KleidiAI kernels may contain them
because KleidiAI picks kernels from the runtime HWCAP probe, not from compile
flags. SVE and SME stay in the exact audited functions and pinned source pairs
of `tools/validate_android_cpu_isa.py`. Each variant must also contain its
declared families outside KleidiAI and export `ggml_backend_score`. LSE atomics
are not audited: GCC's outline-atomic helpers select them at run time.

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
