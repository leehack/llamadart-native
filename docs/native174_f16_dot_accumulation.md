# ARM F16 dot accumulation (#174)

Patch 0005 repairs non-SVE NEON F16 dot products that overflow in F16 before
being returned as F32. Tracking:
[native #174](https://github.com/leehack/llamadart-native/issues/174),
[upstream #31734](https://github.com/ggml-org/llama.cpp/issues/31734)
(open, unfixed in llama.cpp `v0.9.0` and `master` as of 2026-11-07).

## Defect

With `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` and no SVE, upstream
`ggml_vec_dot_f16` maps `GGML_F16_DOT_VEC` to `float16x8_t`: products, partial
sums and the first reduction steps stay in F16 and overflow past 65504. Only
AVX512-FP16 and SVE widen to F32. Shipped `v0.8.0-3` macOS ARM64 and Linux
`linux_armv8.2_2`/`linux_armv8.6_1` modules return `inf` for finite operator
inputs at magnitude 80/90/100; the Linux armv8.0 baseline does not. Ordinary
F16 K attention then produces NaN. Severity is P2: no public-model failure is
established.

## Repair

Only `ggml_vec_dot_f16` changes, in a build-tree copy of
`ggml/src/ggml-cpu/vec.cpp`. Under NEON, FMA, IEEE F16, FP16 vector arithmetic
and no SVE, it multiplies, accumulates and reduces in F32 with four low/high
accumulator pairs (the original 32-element unroll), an eight-element tail and
a scalar tail. When the compiler defines `__ARM_FEATURE_FP16_FML` it uses
FMLAL/FMLAL2; otherwise it converts and uses F32 FMA. ISA flags, runtime
dispatch, ABI, CPU variant selection and cache defaults are unchanged. Other
F16 kernels, including llamafile GEMM, are out of scope.

The repair is longer than the patch policy's "few lines" and the alternative
(disabling FP16 vector arithmetic) is a build workaround that slows every F16
kernel, so it needs the
[narrow exception](../patches/llama.cpp/README.md#proposed-exception-for-native-174)
approved by the maintainer.

## Checks

- `llamadart_f16_dot_accumulation_test` (CTest, `validate_wrapper.yml`) runs
  production graph/backend calls on one thread: the magnitude sweep, widths
  1–129 around every unroll and tail boundary, cancellation, subnormal,
  F16-max and product-overflow inputs, an F32 control, and ordinary and flash
  attention. It reports every failing check. Without the patch it fails on
  Apple silicon with `inf` at magnitude 80/90/100 and NaN attention.
  Zero-length and strided tensors are not covered.
- `tools/linux_cpu_variant_smoke.c` runs the same magnitude sweep on each
  selected Linux arm64 CPU module and rejects nonfinite matmul output.
- The always-active negative-length guard carries the artifact marker checked
  for Android and Apple bundles; it proves inclusion, not correctness.
- The Android ISA audit decodes with `+fp16fml` so FMLAL is not reported as an
  unknown word. Source fingerprints and SVE/SME range checks are unchanged;
  they cover pristine upstream, not the patched copy.

## Before publication

Run the test against each published CPU module (Android variants on device,
Linux variants, Apple) and compare F16-KV prefill/decode throughput with the
previous release. The FHM kernel is slower than the original F16 one in
isolation; no zero-regression claim is made.

Retire the patch when the pinned upstream and every compatibility lane carry a
verified upstream repair; then add its `upstream_fix` text.
