# v0.6.0 Android CPU ISA qualification

Candidate upstream is `d81235049384534c167caea52b85a694f6103d14` (`v0.6.0`),
compared with audited v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`.
The accepted pair binds complete `ggml/src` and KleidiAI `kai` trees:

- ggml/src: `adc8d989599211a4b789b2c595b2ae52335c5a390e6c4972171c46589aa05e7e`
- kai: `64189fc613c1c4c3aaeeb6bb12b38d85dd6728cafd2261a5a88f1b77b10fe59c`

This addition preserves the existing 18 exact ELF function-range allowlist,
source-pair binding, disassembler decoding checks and unknown-source rejection.
It does not move the production submodule or publish a release.

## Source audit

242 files under `ggml/src` changed: 25 CPU, seven shared and 210 accelerator
files. ARM feature scoring (`arch/arm/cpu-feats.cpp`), ARM repack kernels,
KleidiAI kernel selectors/tables (`kleidiai/kernels.*`), its headers and the
remaining selection/compute implementation are unchanged. KleidiAI remains
v1.24.0 with the same archive MD5 and complete `kai` digest.
`kleidiai.cpp` and `repack.cpp` only add null optional buffer allocation/size
callbacks for backend ABI 3; neither changes kernel eligibility or dispatch.

ARM `quants.c` replaces NEON vector initializers with equivalent `vld1`,
`vdup`, `vcreate` and `vcombine` expressions for MSVC compatibility. Existing
DOTPROD/I8MM guards and fallback paths remain; there are no new SVE/SME paths.
CPU CMake adds an MSVC-specific native-probing branch. Android Clang and
Windows ClangCL retain their existing explicit architecture path with
`GGML_NATIVE=OFF`; plain MSVC refuses that unsupported configuration.
The ARM64EC architecture recognition change is restricted to plain MSVC.

The new tiled quantized matmul implementation replaces IQ panel dispatch.
On ARM its `ggml_tiled_supported` returns false because none of AVX, AVX2 or
AVX512VNNI is defined. Both normal and expert matmul entries test that predicate
before dispatch. Its optimized kernels are x86-guarded, so neither the ordinary
nor forced/benchmark path enables a scalable ARM kernel. The new SIMD GEMM
tail kernels are likewise AVX512/AVX2-guarded with a scalar fallback.
`GGML_F16_DOT_*` changes accumulation only under AVX512FP16 and aliases the
old macros on ARM. SpacemiT changes are RISC-V-specific.

Generic CPU changes add BF16 activation/GLU and matmul widening paths, update
workspace sizing for tiled dispatch, and respect reference execution before
llamafile dispatch. They do not add ARM feature-mask bypasses. Shared allocation
changes route through backend ABI-3 tensor-list callbacks with existing linear
allocation fallback. Other shared deltas address quantization range handling,
tensor element-count overflow, graph size arithmetic and GGUF bounds/portability.
These changes do not add intrinsics, inline assembly or optimized ARM dispatch.
This is an ISA containment review, not exhaustive numerical qualification of
all changed operators or unrelated accelerator backends.

## Compiled evidence

The production helper built `android_armv8.2_2` with NDK `28.2.13676358`,
`armv8.2-a+dotprod+fp16`, KleidiAI enabled, `GGML_NATIVE=OFF` and the existing
required feature score. Before adding this pair, the production validator
rejected exactly the source digests above. Its instruction-only scanner, using
the same dynamic-symbol table and disassembler flags as the CLI, inspected
261,286 instructions: all 2,187 SVE/SME instructions were contained in the
existing 18 functions. The unmodified containment policy and public CLI then
passed with the reviewed source pair included.

```bash
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source build/android-v060-isa/_deps/kleidiai-src \
  build/android-v060-isa/bin/libggml-cpu.so
```

The hosted v0.6.0 compiled KleidiAI selector/Q4/Q8 reference checks passed on
the initial PR head in run `37603489552`; they exercise QEMU `cortex-a53` and
`max,sve=off,sme=off`, including unsupported SME requests. Hosted Android
instruction validation and the latest-head platform matrix remain required.
Emulated execution is not physical Android or SME hardware qualification.
No physical-device result, full release packaging or downstream adoption is
claimed by this audit.
