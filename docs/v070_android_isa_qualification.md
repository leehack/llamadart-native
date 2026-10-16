# v0.7.0 Android CPU ISA qualification

Candidate upstream is `620d2225cc436ab0b001e3200e590e3b286ef9ab` (`v0.7.0`),
compared with audited v0.6.0 `d81235049384534c167caea52b85a694f6103d14`.
The candidate uses KleidiAI v1.31.0 instead of v1.24.0. Its complete source
subtree pair is:

- ggml/src: `b571df853210d243a83b1ec34a6936f147916f0369d792f8bb0d17f37fc7894b`
- kai: `deb324dbcf8c952ec149433ab58cbed0b2ab2374a557f0a4eea575b2da761377`

The production submodule, release workflow and downstream pins are unchanged.
Candidate lanes in `validate_wrapper.yml` select the exact SHA separately from
the pinned lanes. Their hosted results are required before merge. This document
records local qualification; it does not claim a published or device-qualified
v0.7.0 runtime.

## Source and dispatch audit

ARM runtime feature detection (`arch/arm/cpu-feats.cpp`), ARM quantization and
repack sources are unchanged. CPU CMake updates KleidiAI's release and archive
MD5; its other change is an s390x architecture spelling. The explicit Android
Clang architecture, isolated CPU variants and required feature scoring remain.

`kernels.cpp` adds a Q4_K table and a direct selector. Its only entry requires
both SME2 and FP16 (mask 48). GEMM uses
`kai_matmul_clamp_f32_f16p4vsx2_qai4c32p16vsx4s1s0sf16_4vsx16vs_sme2_mopa`;
GEMV uses
`kai_matmul_clamp_f32_qsi8d32p1x4_qai4c32p16vsx4s1s0sf16_1x16vs_sme2_dot`.
Their C API wrappers calculate offsets and call the corresponding sized
assembly functions. Their packing and vector-length helpers are reached through
the same selected table; they do not introduce a baseline dispatch path.
The existing Q4_0, Q8_0, F32 and F16 table feature requirements and ordering
remain unchanged.

Q6_K selects the existing Q8_0 kernel family using the same feature subset
checks. Repacking dequantizes Q6_K, computes a row scale and rounds to signed
Q8. It adds no new ISA requirements. Q4_K and Q6_K join the existing workspace,
repacking and quantized-compute paths. Q4_K is disabled unless its table is
selectable; it additionally requires contiguous, unbatched F32 input. GET_ROWS
is removed from the optimized buffer's supported operations. The original
runtime feature probes, SVE width check, SME override validation, selected-kernel
thread cap and fallback-chain feature-mask subtraction remain in place.

KleidiAI v1.31.0 extracts BF16 packing assembly from
`kai_run_lhs_pack_bf16p2vlx2_f32_sme` into
`kai_kernel_lhs_pack_bf16p2vlx2_f32_sme`; the C wrapper calls that sized function.
It remains reachable through the existing SME2 F16 table. Together with the two
Q4_K assembly functions this changes the scalable-instruction ranges from 18
to 20. The validator selects that exact set only for the new source pair.
Earlier pairs retain their original 18 ranges. There is no `kai_*` prefix
exception, and cross-version source pairs or function sets are rejected.

Other CPU deltas concern scalar FP16 SwiGLU, stride-aware clamp, padding and
SET_ROWS threading, x86 quantization/F16C and s390x SIMD. They add no ARM
feature probes, scalable intrinsics or assembly entry points. This review
qualifies ISA containment, not every numerical change or accelerator backend.

## Compiled evidence

The production `build_android_arm64_cpu_variant` helper built
`android_armv8.2_2` with NDK `28.2.13676358`,
`armv8.2-a+dotprod+fp16`, KleidiAI enabled and `GGML_NATIVE=OFF` on macOS arm64.
The unmodified CLI rejected the exact source pair above, reproducing both
failed ARM64 release lanes in
[run 38872836434](https://github.com/leehack/llamadart-native/actions/runs/38872836434).
With the reviewed policy, the same CLI inspected 264,955 instructions:
all 2,542 scalable instructions were contained in the 20 exact functions.

```bash
LLVM="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/darwin-x86_64/bin"
python3 tools/validate_android_cpu_isa.py \
  --objdump "$LLVM/llvm-objdump" --readelf "$LLVM/llvm-readelf" \
  --llama-source third_party/llama.cpp \
  --kleidiai-source build/android-isa-v070/_deps/kleidiai-src \
  build/android-isa-v070/bin/libggml-cpu.so
```

The compiled test links the real CPU backend and checks all 64 masks against
direct selectors and tensor selection, including Q4_K/Q6_K, wrong operations,
missing tensors and incorrect types. Synthetic masks never execute kernels.
The test retains compatibility with old upstream headers through a declaration
probe for the Q4_K selector.

On an Apple M4 Max, with `GGML_KLEIDIAI_SME=0`, all selector checks and 12
matmuls passed: Q4_0/Q8_0/Q6_K through the explicit KleidiAI repacking buffer,
Q4_K through the baseline buffer, with 1, 4 and 9 input columns. Q6_K uses a
256-element row and checks the dequantized scalar reference with a bound for
the documented Q8 rounding; the other existing tolerances are unchanged.
The CPU reports SVE absent and SME present; disabling SME is evidence for the
non-SME path, not execution on a CPU lacking SME hardware.

The same expanded compiled test passed against v0.6.0 with the declaration
probe disabled, including Q4_K/Q6_K baseline fallback. Its rebuilt Android
artifact still passed the original policy: 261,286 instructions, with all
2,187 scalable instructions contained in the original 18 functions.
The Python suite passed 209 tests with one platform-dependent skip. Workflow
lint passed with only the existing `windows-11-vs2026-arm` runner label omitted
from this local actionlint version's outdated label validation.

Hosted qualification passed on implementation head
`c1a51fe7f08819e7f37d223f9fac4312d72ed368` in
[native validation run 38894738711](https://github.com/leehack/llamadart-native/actions/runs/38894738711):
the candidate Android audit, non-SVE QEMU `cortex-a53` and
`max,sve=off,sme=off` compute checks (including unsupported SME requests),
Windows ARM64 builds and the remaining candidate contracts all passed.
See [v070_native_contracts.md](v070_native_contracts.md) for the combined
qualification and consumer handoff. Later PR heads retain their own CI gate.
Physical Android and SME-enabled Q4_K execution are separate qualification
layers and are not claimed here.
