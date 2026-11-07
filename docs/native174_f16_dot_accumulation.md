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
  `ggml_vec_dot_f16` keeps upstream's F16 multiply-adds (four `float16x8_t`
  accumulators over 32-element steps) for at most 64 steps, so 64 products
  per lane, then reduces them in F16 in upstream's order and adds the eight
  lanes into F32. Elements past the last full step add in double, as
  upstream. A nonfinite result is computed again with F32 products and sums
  (FMLAL/FMLAL2 with `__ARM_FEATURE_FP16_FML`, conversion and F32 FMA
  otherwise), which replaces it.
- 0006: the FP16 llamafile tile keeps its F16 multiply-adds, but each lane
  sums at most 64 products (512 K elements) before the sum is added into an
  F32 accumulator (FMLAL/FMLAL2 by one with FP16 FML, conversion and F32 add
  otherwise). An F16 overflow leaves an `inf` or NaN that survives into the
  F32 sum, so a tile with any nonfinite result is computed again by
  `tinyBLAS<4, float32x4_t, ...>` (F16 loads, F32 accumulation), which
  overwrites it.

### Dot contract (0005)

- Overflow: for finite F16 inputs no result is nonfinite. A result whose F16
  sums overflowed, or that had a nonfinite input, comes from the F32
  recompute and is bit-identical to the first version of 0005. NaN and
  infinite inputs still give NaN or infinity.
- Precision: a finite result for `n` < 2080 (one block plus the scalar tail)
  is bit-identical to unpatched upstream. Longer rows add each block's F16
  sums into F32, so F16 rounding no longer compounds across blocks; within a
  block it is upstream's, so results are not F32-accurate.
- Cost: one finiteness check per dot product; a recomputed one costs the F16
  pass plus the F32 pass.

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
  65504 and at 65536 (overflow), lane sums of 2047 that are exact in F16 within
  one block but not with a 65th product (catches a longer block), block tails,
  and overflow confined to single cells of tiles away from the origin, among
  them only the last row and only the last column of a tile (catches a partial
  finiteness check or a misplaced recompute). Weight rows are followed
  by NaN padding, so a read past the row fails; in 0006 the recompute hides an
  over-read that reaches only the padding, and the attention case catches one
  that reaches the next row. It reports every failing check. The macOS
  release lanes also run it built for `armv8.2-a+fp16+dotprod` to cover the
  conversion branch.
- For 0005, on a module with FP16 vector arithmetic and no SVE, the test
  models the patched single-column dot (upstream's F16 multiply-adds,
  reduction order and 64-step block, then F32 lanes and the double tail) and
  requires a bit-exact match wherever the model is finite; elsewhere it
  requires the exact sum. Its 0005 cases: lane sums exact in F16 within one
  block but not with a 65th step (a longer block), a product of 2048 that
  absorbs later products of 1 (F16 rounding, reduction order, a shorter
  block), one overflowing lane in each half, accumulator and block,
  overflows of opposite sign that meet as NaN in the F32 lanes or in the F16
  reduction (a finiteness check that misses NaN), product overflow at widths
  around the recompute's eight-element and scalar tails, rows without NaN
  padding (an over-read the recompute would hide), and NaN and infinite
  inputs.
- On Apple silicon (FMLAL and conversion builds) it fails 32 checks without
  0005, 19 with the first 0005 (bit-identity only), and 2 to 43 with each of
  23 mutants of the new 0005: no flush, blocks of 32, 63, 65 or 128 steps, a
  flush that overwrites instead of adding, no recompute, recompute always,
  finiteness checks of infinity, NaN, the low half or one lane only, a
  dropped scalar tail, partial block or high half, over-reads, a reordered
  F16 reduction or a dropped accumulator, and a recompute that skips a tail,
  drops its high half or returns NaN. Moving the recompute's eight-element
  loop into its scalar tail is equivalent and survives. Without
  0006's flush, recompute or tail block it failed 1 to 202 checks when 0006
  landed; those GEMM cases are unchanged. Zero-length and strided query
  tensors are not covered.
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
`v0.9.0` release, which a maintainer dispatches manually once the v0.9.0 pin
is on `main`:

- Run the test against each published CPU module: Android variants on a
  device, Linux variants, Apple.
- Compare prefill/decode throughput on an Android device for each variant with
  the previous release, F16 KV with FA off and on, and an F16-weight model on
  `linux_armv8.2_2`.

Retire a patch when the pinned upstream and every compatibility lane carry a
verified upstream repair; then add its `upstream_fix` text.
