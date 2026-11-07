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
- 0006: the FP16 llamafile tile keeps its F16 multiply-adds, but each lane
  sums at most 64 products (512 K elements) before the sum is added into an
  F32 accumulator (FMLAL/FMLAL2 by one with FP16 FML, conversion and F32 add
  otherwise). An F16 overflow leaves an `inf` or NaN that survives into the
  F32 sum, so a tile with any nonfinite result is computed again by
  `tinyBLAS<4, float32x4_t, ...>` (F16 loads, F32 accumulation), which
  overwrites it.

### GEMM contract (0006)

- Overflow: for finite F16 inputs no output is nonfinite. A tile that
  overflowed in F16, or had a nonfinite input, is bit-identical to the F32
  tile that the first version of 0006 used for every tile.
- Precision: a tile that needs no recompute sums every 512 K elements in F16,
  in upstream's order. For `k` ≤ 512 its result is bit-identical to the
  unpatched tile; for longer rows the F16 rounding no longer compounds across
  blocks. It is not F32-accurate: each multiply-add rounds to F16, so results
  below the F16 subnormal range become zero.
- Cost: a redone tile costs the F16 pass plus the F32 pass. With every tile
  overflowing, F16 GEMM takes 1.4 times as long as with the F32 tile alone.

ISA flags, runtime dispatch, ABI, CPU variant selection and cache defaults are
unchanged. Both patches exceed the policy's "few lines" condition; the
maintainer approved the
[narrow exception](../patches/llama.cpp/README.md#exception-for-native-174).

## Performance

Apple M4 Max, CPU only, `GGML_NATIVE=OFF`. Kernel costs are against the
unpatched kernels; model throughput is against the shipped `v0.8.0-3` macOS
library (Qwen2.5-0.5B-Instruct, 8 threads, F16 KV, median of 3 interleaved
rounds' ratios):

| Case | Cost |
| --- | --- |
| F16 mat-vec 4096x4096, 1 thread (0005): FMLAL / conversion path | +13% / +78% time |
| F16 GEMM 4096x4096, 32–512 columns, 1 or 8 threads (0006): FMLAL / conversion flush | +8–13% / +12–14% time |
| Same with every tile overflowing (all redone in F32) | +257% time |
| F16-weight model pp512, FA off / on | 0.91 / 0.94 tokens/s |
| F16-weight model tg128 and tg64 at depth 1024 (decode does not reach the tile) | 0.96–1.02 (noise) |
| Q8_0 model pp512, FA off / on | 0.98 / 1.01 tokens/s |
| Qwen3-TTS cancel latency (core `tts` pack, CPU), median / p90 / max | 16 / 40 / 49 ms against 15 / 32 / 48 ms |

The first version of 0006 accumulated every tile in F32: F16 GEMM +125–163%
time, F16-weight pp512 at 0.44 of `v0.8.0-3` and a median TTS cancel latency
of 65 ms. Blocks of 16 or 32 products per lane, instead of 64, gave pp512
at 0.79 or 0.87. `linux_armv8.2_2` runs the conversion flush; it is not yet
measured on that hardware.

## Checks

- `llamadart_f16_dot_accumulation_test` (CTest, `validate_wrapper.yml`) runs
  production graph/backend calls on one thread: the magnitude sweep, widths
  1–129 around every unroll and tail boundary, cancellation, subnormal,
  F16-max and product-overflow inputs, a cancellation whose F16 partial sums
  overflow, 8/16/32-column GEMM cases, F32 controls, and ordinary and flash
  attention over 16 keys for one and eight queries with an unsaturated
  softmax. For 0006 it adds GEMM blocks whose F16 lane sums end exactly at
  65504 and at 65536 (overflow), sums that are exact in F16 within one block
  but not across two (catches a longer or missing block), block tails, and an
  overflow confined to part of one tile away from the origin (catches a
  partial finiteness check or a misplaced recompute). Weight rows are followed
  by NaN padding, so a read past the row fails; in 0006 the recompute hides an
  over-read that reaches only the padding, and the attention case catches one
  that reaches the next row. It reports every failing check. The macOS
  release lanes also run it built for `armv8.2-a+fp16+dotprod` to cover the
  conversion branch.
- Without the patches it fails on Apple silicon: 16 checks with 0005 alone
  (GEMM overflow and block cases, eight-query ordinary attention); without
  0006's flush, recompute or tail block, 1 to 202 checks.
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
