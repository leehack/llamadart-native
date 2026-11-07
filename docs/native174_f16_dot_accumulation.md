# ARM F16 dot accumulation candidate (#174)

This source candidate repairs non-SVE NEON F16 dot products that overflow in
F16 before being returned as F32. It changes only `ggml_vec_dot_f16` in a
build-tree copy of upstream `ggml/src/ggml-cpu/vec.cpp`. Multiplication,
accumulation and reduction use F32; four low/high accumulator pairs preserve
the 32-element unroll, with full eight-element and scalar tails.

Tracking: [native #174](https://github.com/leehack/llamadart-native/issues/174),
[upstream #31734](https://github.com/ggml-org/llama.cpp/issues/31734).
The base is owner `05ef9a48a2bd1db2f4208a89c393794412e9cb74`, pinned upstream
`8642541b255bfb5dc39cd351a178be09013a55d6`.

## Evidence and scope

ROOT reproduced `inf` with success status for the magnitude 80/90/100 F16
operator cases against actual public `v0.8.0-3` macOS ARM64 and Linux ARM64
`linux_armv8.2_2`/`linux_armv8.6_1` modules. The Linux baseline passes all
11 cases. These are model-free operator results, not public-model/app
failures or proof of a new release regression. Severity remains P2.

The carried branch requires NEON, FMA, IEEE F16, FP16 vector arithmetic and
no SVE. When the compiler defines `__ARM_FEATURE_FP16_FML`, it uses
FMLAL/FMLAL2 to accumulate the low/high F16 halves directly into F32.
Otherwise it retains the conversion/F32-FMA path. The fast path uses only
the existing compiler-advertised capability; it adds no ISA flags or runtime
dispatch. Global SIMD mappings, ABI, CPU variant selection and cache defaults
are unchanged. Other F16 kernels, including separate llamafile GEMM, are
outside the repair.

Patch 0005 requests the explicit, issue-specific policy exception described
in [the patch policy](../patches/llama.cpp/README.md#proposed-exception-for-native-174).
Disabling optimized FP16 is a broad build workaround; this internal correctness
repair is more than a few lines. Neither original eligibility condition is
claimed satisfied. Acceptance requires explicit maintainer approval of this
exception in the PR; publication is a separate decision.

## Durable checks and remaining gates

`llamadart_f16_dot_accumulation_test` uses production ggml graph/backend APIs,
with one CPU thread and checks that remain active in Release. It covers
operator magnitudes, vector/tail boundaries and synthetic ordinary/flash
attention. Run it against each exact candidate/control platform before
claiming corrected behavior there. A candidate pass alone does not establish
causal attention failure/restoration. Zero-length kernel execution and
noncontiguous/strided tensor paths are not covered by this candidate test;
the width sweep begins at one.
The Linux compatibility smoke retains its original matrix shape and rejects
nonfinite results/errors explicitly.

Local M4 Max CPU source builds pass the full numerical test for both the
conversion-only `5fcce7c` and FHM `acebd230` candidates; the original control
fails. With one hash-bound cached GGUF, the conversion-only build's median
decode throughput was 1.6–1.9% lower than the original, with FA off/on. This
does not measure FHM model performance, model quality or an affected public
model.

Across 27 rotated direct-dot runs at lengths 64/128/256, the FHM build's median
latency was 27–46% lower than conversion-only, but still 19–25% higher than the
original. The first length-64 run had a broad outlier; retain the full ranges
and provenance in the PR evidence. These kernel measurements do not establish
FHM whole-model performance or other-platform qualification.

Source-only Python patch checks cover exact application, markers, mismatch
skip/strict failure and pristine upstream preservation. Android and Apple
artifact markers retain the patch in release validation. The always-active
negative-length guard carries the marker; its presence proves inclusion,
not numerical correctness. Verify the marker in every actual optimized and
baseline module, including Linux variants.

The Android ISA audit enables `+fp16fml` in objdump so FHM instructions decode
instead of appearing as unknown words. This changes decoder selection only;
source fingerprints and exact SVE/SME function-range checks stay unchanged.

Before merge or publication, ROOT must provide:

- Paired original/candidate/reverted builds with identical source and flags,
  finite/reference checks and exact module identities for the frozen 11-case
  probe and the durable numeric test. Preserve original failures as evidence.
- Actual selected Linux/Android variant correctness and ISA containment;
  SVE, iOS and Windows remain separate platform gates. Existing ISA source
  fingerprints cover pristine upstream, so audit compiled patched copies too.
- Paired timing for dot lengths 64/128/256 and CPU prefill/decode with F16 KV,
  FA off/on, using identical available public model bytes, threads and inputs.
  F32 widening can slow the hot loop; no zero-regression claim is made.
- An independent review of the exact candidate revision and explicit approval
  of the narrow patch-policy exception. No owner release or core/Apple pin
  adoption follows automatically.

Remove this patch only when the pinned upstream and every compatibility lane
contain a verified upstream repair, or the defect no longer reproduces without
it. No `upstream_fix` string is invented before such a repair exists.
