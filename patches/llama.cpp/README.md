# Carried llama.cpp patches

Changes to llama.cpp sources that this repository's builds apply on top of the
pinned upstream commit. `third_party/llama.cpp` itself stays exactly as
upstream published it.

Updating the submodule to an upstream release that has the fix is still the
preferred way to get a fix (`AGENTS.md`). A patch is carried only when all of
these hold:

- A released artifact crashes or gives wrong results, and nothing in `src/` or
  the build configuration can prevent it.
- No upstream release has the fix, and the fix is a few lines.
- The patch has an issue in `leehack/llamadart-native` or `leehack/llamadart`
  and, where one exists, an upstream pull request or issue.

## Exception for native #174

The maintainer approved a narrow exception to the first two eligibility
conditions above for patches 0005 and 0006 only. Disabling FP16 vector
arithmetic or llamafile is a build workaround that slows unrelated kernels
(llamafile off costs Q4_0/Q8_0 prefill too), and the F16 dot repair is more
than a few lines.

Published ARM64 artifacts return nonfinite results for finite F16 dot
products. Patch 0005 makes the non-SVE NEON `ggml_vec_dot_f16` add its F16
sums into F32 every 2048 elements (64 products per lane) and recompute in F32
any result that is nonfinite; patch 0006 does the same for the FP16 llamafile
GEMM tile every 512 K elements. Neither changes ISA
flags, ABI, variants or cache defaults. Performance cost, checks and
publication gates are in
[native174_f16_dot_accumulation.md](../../docs/native174_f16_dot_accumulation.md).
This exception does not extend eligibility to other patches.

## Exception for native #192

The maintainer approved (2026-11-17) a second narrow exception to the first two
eligibility conditions above for patches 0008 and 0009 only. They repair a
throughput regression, not a crash or wrong result, and are about 25 and 8
lines.

[Upstream #31854](https://github.com/ggml-org/llama.cpp/pull/31854) took F16
`src0` matmuls and the flash-attention K.Q dot off the F16 x F16 kernels; on ARM
with FP16 vector arithmetic they run `ggml_vec_dot_f16_f32` instead, 0.16 to
0.6 of `v0.9.0` for F16 weights on Neoverse-N1. Patch 0008 returns an F16
`GGML_OP_MUL_MAT` with F32 `src1` and no `src1` precision request (or
`GGML_PREC_F16`) to the F16 route of `v0.9.0`: `src1` rounded to F16, so 0005
and 0006 apply again. An explicit F32, BF16, Q8 or Q4 `src1` request and the
`CONV_2D`/`CONV_3D` matmuls keep upstream's F32 route. Patch 0009 does the same for an F16 K in the
flash-attention one-chunk kernel. x86 and modules without FP16 vector
arithmetic are unchanged; Windows arm64 builds without FP16 vector arithmetic
(and ClangCL defines `_MSC_VER`), so it is unchanged. Results on ARM FP16 modules then differ from upstream's
CPU reference by the query rounding that `v0.9.0` had: unhinted F16 matmul
inputs above 65504 become `inf` and values below about 3e-8 flush to 0, the
same as `v0.9.0` and new relative to upstream v1.0. Evidence,
checks and gates:
[native192_arm_fp16_f16_route.md](../../docs/native192_arm_fp16_f16_route.md).
This exception does not extend eligibility to other patches.

Remove both when upstream ships a `src1`-precision-gated F16 x F16 path for the
CPU backend, or ARM F16 x F32 kernels, that keep F16 prefill and decode at
0.97 or more of `v0.9.0` on the FP16 modules.

## Files

- `series.json` lists the patches in the order they apply. Each entry has the
  patch `file`, the `tracking` issue, the `upstream` pull request or issue to
  watch (a permalink to the upstream lines when there is none), and
  `android_markers`: a `library` of the Android bundle and a `text` that only
  the patched source compiles into it. A patch of an Apple-only source has
  `apple_markers` instead, for the Apple bundles. An optional `upstream_fix`
  is exact text that the upstream repair puts in the patched source, or in the
  file named by `upstream_fix_path` when it repairs another one; see "Retiring
  a patch upstream has fixed".
- `NNNN-short-name.patch` is a unified diff (`git diff -U2` inside
  `third_party/llama.cpp`) of one existing `.c`, `.cpp`, `.m` or `.mm` file. A
  patch cannot add a file or edit a header: only a source that an upstream
  target compiles can be replaced.

## How they are applied

`cmake/llama_cpp_patches.cmake` runs `tools/llama_cpp_patches.py` while CMake
configures. It writes the patched copy of each source to
`<build>/llama.cpp-patched/` and makes the upstream target compile that copy
instead of the original.

A hunk applies only where its context and removed lines occur exactly once in
the upstream file. There is no fuzz. A patch that does not fit the llama.cpp
being built is skipped whole with a CMake warning that names it. The bundle
marker checks below catch missing patches.

## What keeps a release from losing a patch

`tools/validate_android_artifacts.py` fails an Android bundle in which a
library lacks the `text` of an `android_markers` entry. It reads the built
library, so it does not matter why the patch is missing. A marker on
`libggml-cpu.so` is checked in every `libggml-cpu*.so` of the bundle (Android
arm64 ships only the `libggml-cpu-android_*.so` variants). Android core bundles
must contain CPU modules even when all CPU marker patches are retired. The
backend-only OpenCL release lanes skip CPU packaging checks because their
payload contains only the OpenCL library. The Linux arm64 bundle audit
(`tools/check_linux_arm64_cpu_variants.sh`) checks the same CPU markers in the
baseline and each shipped variant.

The release and `validate_wrapper.yml` run these checks on their bundles, so
a patch that upstream has outgrown fails the pull request that moves the
upstream, or the manually dispatched release (`native_release.yml`, or
`auto_native_release.yml` for a new stable upstream tag). Check then
whether upstream fixed the defect, and refresh the patch or delete it.

For that reason every patch must have at least one marker, and so must change
a source that an Android library is built from, or an Apple-only source whose
`apple_markers` `tools/llama_cpp_patches.py check-apple` looks for in every
Apple bundle. A patch fits the same upstream source on every platform or on
none, so a release whose Android bundle has every Android marker has those
patches on its other platforms too.

A patch cannot change a `.metal` kernel: the upstream build embeds those files
through its own custom commands, which take no replacement source. A patch
that edits embedded text at runtime can use `required_text` to verify the
original text is present. A patch whose specified path lacks that text does
not fit, which drops its marker and fails `check-apple`.

`python3 tools/llama_cpp_patches.py apply --strict --upstream
third_party/llama.cpp --output <dir>` fails where the build would skip.

## Retiring a patch upstream has fixed

An entry with `upstream_fix` is skipped when llama.cpp contains that fix.
Its markers are then optional and the manifest omits it. Pick text unique
to the fix so builds of unfixed sources still require the patch.

## What a release records

`assets.json` has `llama_cpp_patches`: the `file`, `sha256` and `tracking` of
every patch that the released llama.cpp needs, an empty list when there is
none. `llama_cpp_commit` remains the upstream commit the patches apply to.

## Adding, refreshing or removing a patch

1. Edit the source inside `third_party/llama.cpp`, write
   `git -C third_party/llama.cpp diff -U2 > patches/llama.cpp/NNNN-name.patch`,
   then restore the submodule with `git -C third_party/llama.cpp checkout -- .`.
2. Add or remove the `series.json` entry.
3. Run `python3 -m unittest tests.test_llama_cpp_patches` with the submodule
   checked out, and build a target that compiles the patched source.
4. Describe the patch, its evidence and what removes it in the document of the
   rebuild that introduces it under `docs/`.

Remove a patch when the pinned upstream in the sole `LLAMADART_UPSTREAMS`
row of `validate_wrapper.yml` contains the fix, or the defect no longer
reproduces without it.
