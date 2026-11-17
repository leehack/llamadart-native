# v1.0.0 ARM source and ISA audit

The fixed candidate is llama.cpp
`b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (resolved stable v1.0.0), compared
with v0.9.0 `cd959bae99ff4bed22bef2604f45241c09bbb8c8`. KleidiAI remains
v1.32.0, release archive MD5 `05f1953428ffe8e8c7f585dc6d508887`. The reviewed
complete unpatched source pair is:

- ggml/src: `9eb33398797f3bba473032917ebe0cd017a67ff12eeaf3d88e4c2552bbfa94e1`
- kai: `fd5b9ca98d07b81671eca377c258da74e72173f92f3cab0acdf6c17ab870bd9d`

## Source review

`ggml/src/ggml-cpu/arch/arm/`, the entire `ggml-cpu/kleidiai/` tree,
`ggml-cpu-impl.h` and CPU CMake are byte-identical to v0.9.0. This preserves
HWCAP/SVE256/SVE-I8MM/SME probes, kernel-table masks and order, vector-length
helpers, SME overrides, fallback-mask subtraction, tensor admission, restored
Windows C-kernel references and per-source ISA flags. The actual cached release
archive is checksum-verified before use; its `kai` digest matches the existing
v0.9 pair. No source is borrowed from a mutable branch.

The twelve changed CPU files add no ARM feature probe or Kleidi caller:

- `ggml-cpu.c/.cpp` route F16 matmul and matmul-ID queries through F32 and
  widen F16/BF16 inputs; removed ADD1/DIAG_MASK ops and workspace updates match
  the public enum pruning.
- `vec.cpp/.h` add F16-to-F32 dot/MAD. ARM uses existing F32 SIMD/dot and
  NEON widening. Any SVE in the existing F32 dot is still compile-flag gated;
  the release non-SVE variant does not define that flag. The old F16 kernel
  remains and still receives carry 0005.
- `ops.cpp/.h` remove old ops, propagate RoPE truncation, retain F32 convolution
  inputs, and replace the CPU-FA F16 V accumulator with F32. They add no new
  scalable intrinsic/assembly entry or unguarded Kleidi call. Numerical and
  real-model correctness are separate gates.
- `llamafile/sgemm.cpp` adds F16 x F32 support for x86, s390 vector work and
  associated helpers; the ARM F16 x F16 branch remains, with carry 0006.
- `simd-gemm.h` corrects `__ARM_NEON__` to `__ARM_NEON` for tile sizes.
- Four `spacemit/` files are RISC-V IME/RVV work, excluded from the ARM target.

The rest of the fingerprinted ggml tree includes accelerator/backend changes;
none changes the source of the standalone ARM CPU module's feature dispatch.
The source comparison, checksums and raw diff receipts are retained in the
implementation evidence directory named in [v100_qualification.md](v100_qualification.md).

## Compiled controls before source admission

2026-11-15, Apple M4 Max/macOS arm64, AppleClang 21.0.0:
`llamadart_kleidiai_dispatch_test` links the actual candidate CPU library and
passes all **128 feature masks**, positive/negative tensor paths and **12**
Q4_0/Q8_0/Q4_K/Q6_K matrix/scalar-reference comparisons with 1, 4 and 9 columns.
`GGML_KLEIDIAI_SME=0` selects the non-SME control. Hardware reports SVE absent,
SME/SME2 present; this is not execution on hardware lacking SME, nor SVE-kernel
execution. Q4_K takes baseline fallback in this control.

The production `build_android_arm64_cpu_variant` helper built
`android_armv8.2_2`, `armv8.2-a+dotprod+fp16`, Kleidi on, native off,
llamafile off, Android API28, pinned NDK **28.2.13676358** on macOS arm64.
Only the FetchContent source location was overridden to the checksum-verified
v1.32.0 archive extracted inside the evidence directory. No toolchain was
installed or changed. The exact module SHA-256 is
`999dab72022ec30a7a2214bc1c4ca9fb14213c90c678f4bfaacf93fd2c4d9567`.

The unchanged validator first rejects that real module with the unaudited
source-pair error. After source review and the compiled controls above, adding
only this exact pair passes: **266078 instructions**, **3130 SVE/SME instructions
inside the same 22 exact audited functions** as v0.9.0. No function-range or
instruction-family exception is added. Both previous pairs and unknown,
cross-combined, added-caller, changed-kernel and escaped-instruction negative
controls remain in `tests/test_android_cpu_isa.py`.

```sh
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source <evidence>/kleidiai-v1.32.0 \
  <evidence>/build-v1-android/bin/libggml-cpu.so
```

This admits source identity and establishes bounded compiled containment. It
does not qualify Android execution, a published archive or every platform.
PR gates must still pass at the final head: Windows ARM64 ClangCL, Android
artifact/disassembly, non-SVE QEMU selectors/scalar-reference compute, Linux
variants/glibc/packaging and the separate ggml SVE256 paths. Physical provider
performance and original numerical/model/device triggers remain separately
approved qualification in native #112/#145/#187.
