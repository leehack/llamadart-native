# ARM F16 dot accumulation (#174)

Patches 0005 and 0006 repair non-SVE ARM F16 dot products that overflow in
F16 before being returned as F32. Tracking:
[native #174](https://github.com/leehack/llamadart-native/issues/174),
[upstream #31734](https://github.com/ggml-org/llama.cpp/issues/31734)
(open, unfixed in llama.cpp `v0.9.0` and `master` as of 2026-11-07).
Published artifacts stay affected until a release carrying both patches
replaces them.

## Defect

With `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (macOS arm64 and the arm64 iOS
simulator by default, `+fp16` Linux and Android variants; not iOS devices,
whose default target lacks it), two CPU kernels multiply and sum F16 inputs in
F16 and overflow past 65504:

- `ggml_vec_dot_f16` (one query column) maps `GGML_F16_DOT_VEC` to
  `float16x8_t` unless the build has SVE. Only AVX512-FP16 and SVE widen.
- llamafile `tinyBLAS<8, float16x8_t, ...>` (eight or more query columns,
  `llamafile_sgemm`) accumulates `float16x8_t` tiles. ggml builds it with
  `GGML_LLAMAFILE`, which Android turns off and which ggml compiles out under
  SVE or I8MM, so it reaches macOS arm64 and Linux `linux_armv8.2_2`.

Shipped `v0.8.0-3` macOS ARM64 and Linux `linux_armv8.2_2`/`linux_armv8.6_1`
modules return `inf` for finite operator inputs at magnitude 80/90/100; the
Linux armv8.0 baseline does not. Ordinary F16 K attention then produces NaN,
for one query and, through GEMM, for eight or more. Severity is P2: no
public-model failure is established.

## Repair

Both patches change build-tree copies only.

- 0005: under NEON, FMA, IEEE F16, FP16 vector arithmetic and no SVE,
  `ggml_vec_dot_f16` multiplies, accumulates and reduces in F32 with four
  low/high accumulator pairs (the original 32-element unroll), an eight-element
  tail and a scalar tail. With `__ARM_FEATURE_FP16_FML` it uses FMLAL/FMLAL2;
  otherwise it converts and uses F32 FMA.
- 0006: the FP16 branch of `llamafile_sgemm` instantiates
  `tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, float>`,
  which loads F16 and accumulates in F32, instead of the F16 tile.

ISA flags, runtime dispatch, ABI, CPU variant selection and cache defaults are
unchanged. Both patches exceed the policy's "few lines" condition; the
maintainer approved the
[narrow exception](../patches/llama.cpp/README.md#exception-for-native-174).

## Performance

Apple M4 Max, CPU only, `GGML_NATIVE=OFF`, against the original kernels:

| Case | Cost |
| --- | --- |
| F16 mat-vec 4096x4096, 1 thread: FMLAL / conversion path | +13% / +78% time |
| F16 GEMM 4096x4096, 32–512 columns, 1 or 8 threads (0006) | +125–160% time |
| Same with `GGML_LLAMAFILE=OFF` instead of 0006 | +201–248% time; Q4_0/Q8_0 GEMM +50–58% |
| Llama-3.2-1B Q4_K_M pp512, F16 KV, 8 threads, FA off / on | -4.7% / -1.5% tokens/s |
| Same, 1 thread, FA off | -4.0% tokens/s |
| LFM2-VL-450M Q4_0 pp512, F16 KV, 8 threads, FA off | -3.0% tokens/s |
| Same models with `GGML_LLAMAFILE=OFF` instead of 0006, FA off | -7.4% to -9.1% tokens/s |

F16-weight models pay the GEMM cost on most of their prefill: the
`linux_armv8.2_2` F16 prefill gain over the armv8.0 baseline came from the F16
tile and is expected to largely go away (not yet measured there). A widening FMLAL tile was slower than the
conversion tile (+270% time) because its accumulators exceed the register file.

## Checks

- `llamadart_f16_dot_accumulation_test` (CTest, `validate_wrapper.yml`) runs
  production graph/backend calls on one thread: the magnitude sweep, widths
  1–129 around every unroll and tail boundary, cancellation, subnormal,
  F16-max and product-overflow inputs, a cancellation whose F16 partial sums
  overflow, 8/16/32-column GEMM cases, F32 controls, and ordinary and flash
  attention over 16 keys for one and eight queries with an unsaturated
  softmax. Weight rows are followed by NaN padding, so a read past the row
  fails. It reports every failing check. The macOS release lanes also run it
  built for `armv8.2-a+fp16+dotprod` to cover the conversion branch.
- Without the patches it fails on Apple silicon: 30 checks with neither, and
  the seven GEMM checks plus eight-query ordinary attention with 0005 alone.
  A `n - n % 4` tail mutation of 0005 fails 15 checks through the padding. Zero-length and
  strided query tensors are not covered.
- `tools/linux_cpu_variant_smoke.c` runs the single-column magnitude sweep on
  each selected Linux arm64 CPU module and rejects nonfinite matmul output.
- Always-active guards (`n >= 0` in 0005, `k >= 0` in 0006) carry the
  artifact markers: Android and Apple for 0005, Apple for 0006 (Android builds
  no llamafile). They prove inclusion, not correctness.
- The Android ISA audit decodes with `+fp16fml` so FMLAL is not reported as an
  unknown word. Source fingerprints and SVE/SME range checks are unchanged;
  they cover pristine upstream, not the patched copies.

## Before publication

These gates hold before any release carries the patches, including the
`v0.9.0` release that the daily stable detector publishes once its ISA audit
passes on `main`:

- Run the test against each published CPU module: Android variants on a
  device, Linux variants, Apple.
- Compare prefill/decode throughput on an Android device for each variant with
  the previous release, F16 KV with FA off and on, and an F16-weight model on
  `linux_armv8.2_2`.

Retire a patch when the pinned upstream and every compatibility lane carry a
verified upstream repair; then add its `upstream_fix` text.
