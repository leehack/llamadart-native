# ARM FP16 F16 matmul route (#192)

Owner: [native #192](https://github.com/leehack/llamadart-native/issues/192).
Patches 0008 and 0009 are stacked on the llama.cpp v1.0.0 refresh
([#188](https://github.com/leehack/llamadart-native/pull/188)) and exist only
while upstream has no equivalent. The maintainer chose this option on
2026-11-17 and approved the
[patch-policy exception](../patches/llama.cpp/README.md#exception-for-native-192).

## Defect

[Upstream #31854](https://github.com/ggml-org/llama.cpp/pull/31854) (`12ab5979`)
makes the CPU trait of F16 `vec_dot_type = F32` and `vec_dot =
ggml_vec_dot_f16_f32`, "a stopgap" outside AVX that converts 64 elements to a
stack buffer and calls the F32 dot. On ARM with FP16 vector arithmetic:

- `llamafile_sgemm` has no `Btype == F32` case in that branch, so the FP16 tile
  (0006) is never reached and each F16 weight cell is one stopgap dot.
- Mat-vec loses `ggml_vec_dot_f16` (0005) and its 8 F16 lanes per FMA.
- The flash-attention one-chunk kernel takes its K.Q dot from the same trait.

Against `v0.9.0` on Neoverse-N1 (`linux_armv8.2_2`, t2a): F16 pp512 0.16, F16
tg128 0.58–0.60, Q8_0 pp512 with F16 KV and FA off 0.54. Android modules have no
llamafile, so they lose the dot-against-dot route (about 0.57–0.83). Upstream
documents the cost as accepted for now and measured ARM only on an M5 Max.

## Carry

| Patch | Source | Change |
| --- | --- | --- |
| 0008 | `ggml-cpu.c` | `ggml_mul_mat_src1_as_f16(dst)`: on `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (not MSVC), a `GGML_OP_MUL_MAT` with F16 `src0`, F32 `src1` and `ggml_prec_get_src(dst, 1)` undefined or F16 uses `vec_dot_type = F16` and `ggml_vec_dot_f16`, in the work-buffer size, the conversion and the chunk kernel. The direct `llamafile_sgemm` call is skipped for it so the F16 tile runs on the converted `src1`. |
| 0009 | `ops.cpp` | The flash-attention one-chunk kernel takes `ggml_vec_dot_f16` with Q rounded to F16 for an F16 K, on the same guard. |

Markers `llamadart F16 matmul route` and `llamadart F16 K.Q route` sit in a
`GGML_ASSERT` outside the `#if`, so every CPU module and the Apple library
carries them.

- The `dst->op == GGML_OP_MUL_MAT` guard matters: `CONV_2D` calls the matmul on
  a tensor without an op whose `src1` is the op's own work buffer, and an F16
  route there gives inconsistent results on several threads.
- An explicit `GGML_PREC_F32` or `GGML_PREC_BF16` request on `src1` is honored,
  which `v0.9.0` ignored.
- Not covered: `GGML_OP_MUL_MAT_ID` (F16 experts stay on upstream's route) and
  F16 `src0` with F16 `src1`.
- The guard also puts the Android SVE2 module on its SVE `ggml_vec_dot_f16`
  again. That module was not built or run for this change.
- x86 and modules without FP16 vector arithmetic are unchanged.

Numerics: ARM FP16 CPU graphs return to `v0.9.0` (query rounded to F16, 0005 and
0006 live). They differ from upstream's CPU reference by that rounding until
upstream ships its own gated path.

## Remove when

Upstream ships a `src1`-precision-gated F16 x F16 path in the CPU backend (the
TODO in `ggml_compute_forward_mul_mat` names it), or ARM F16 x F32 kernels, with
F16 prefill and decode at 0.97 or more of `v0.9.0` on the FP16 modules. Then add
`upstream_fix` text or delete the patches.

## Evidence

Local throughput (Apple M4 Max, shared host, interleaved medians of 3–5
rounds), ratio to `v0.9.0` with 0005/0006, Qwen2.5-0.5B:

| Build | F16 pp512 | F16 tg128 | Q8_0 pp512, FA off | Q8_0 decode at depth, FA off / on |
| --- | --- | --- | --- | --- |
| `b1b770d9` | 0.25–0.28 | 0.72–0.83 | 0.74 | 0.74–0.89 / 0.79–0.90 |
| + 0008 + 0009 | 0.98–1.01 | 0.98–0.99 | 1.00–1.01 | 0.91–0.95 / 0.87–0.90 |
| Docker GCC 13.3 `armv8.2-a+dotprod+fp16`, + both | 1.00–1.04 | 1.03 | 1.01 | 0.97 / 0.91 |
| Docker GCC 13.3 `armv8.6-a+...+i8mm` (no llamafile), + both | 1.00–1.04 | 1.01–1.06 | not run | 1.01 / 0.875 |

The FA-on depth-decode loss that remains is upstream `7f24aacd` (the nonfinite
V accumulator fix, [native #187](https://github.com/leehack/llamadart-native/issues/187)),
not this route.

`llamadart_f16_dot_accumulation_test` follows the route: an unhinted F16 x F32
graph rounds the query to F16 on FP16 vector arithmetic modules and keeps it F32
elsewhere on upstream v1. New graph rows:

| Row | Expected |
| --- | --- |
| `route-unhinted`, 1 and 16 columns, F16 ones times `1 + 2^-12` | 256 on the F16 route, 256.0625 otherwise |
| `route-src1-f32`, `route-src1-bf16` | 256.0625 (256 on the F16 trait) |
| `conv-route`, `ggml_conv_2d_direct`, F16 kernel, 4 threads | 128.03125 in every element |
| `flash-route`, keys +64 and −64, DK 64, query `1 + 2^-12`, V (1, −1) | `tanh(4096 s)` on the F16 route, `tanh(4097 s)` otherwise, `s = 1/8192` |

The test passes in graph and `--residual` modes on macOS arm64 (conversion-path
build, and the shipped `macos-arm64-full` preset with FMLAL) and on Linux arm64
(Ubuntu 24.04, GCC 13.3, `armv8.2-a+dotprod+fp16` and
`armv8.6-a+dotprod+fp16+i8mm`; the latter has no llamafile, so its `--residual`
GEMM cases are not executed). `libggml-cpu.so` and the Apple library carry both
markers.

Mutations, each rebuilt on macOS arm64 (conversion-path build):

| Mutation | Result |
| --- | --- |
| Drop `dst->op == GGML_OP_MUL_MAT` from 0008 | `conv-route` fails (80.3 against 128.03125) |
| Drop the `prec` condition from 0008 | `route-src1-f32` and `route-src1-bf16` fail, 4 checks |
| Remove 0008 from the series | `route-unhinted` and the F16 oracle rows fail; `check-apple` fails on the built library |
| Remove 0009 from the series | `flash-route` fails; `check-apple` fails on the built library |

## Gates

Each of these needs the PR head:

- CI, independent audit (native kernel change), and the Android ISA audit for
  the changed `ggml-cpu.c` and `ops.cpp`.
- t2a and c4a throughput: `cand/pub090` at 0.97 or more outside the FA-on
  depth-decode cells.
- The test on every published CPU module, the Nemotron CPU-FA rows, ON_DEVICE CPU
  rows, CPU smoke, the S24 and A53 C1 and S2 reruns with new APKs, a Qwen3-TTS
  cancel-latency check and the FA-off cross-layout NMSE gate.
- The throughput figures above are local M4 Max runs on a shared host, not N1,
  V2, Android or x86 numbers.
