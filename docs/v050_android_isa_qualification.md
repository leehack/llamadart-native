# v0.5.0 Android CPU ISA qualification

The Android arm64 OpenCL build in native release run `35976011426` compiled
successfully, then rejected the new complete `ggml/src` fingerprint. It
selected upstream v0.5.0 at `7fe450e19305b828c199d602c23a8337aaa1f03b`. The
arm64 Vulkan job in the same run failed earlier, in shader generation, so it
never reached this gate. This policy update does not change the production
submodule pin, the release channel, or the dispatched-function allowlist.

## Source review

Compared with the previously audited v0.4.1 tree
(`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`), 203 files under `ggml/src`
changed: eight CPU files, five shared files, and 190 accelerator files
(Hexagon, OpenVINO, Vulkan, Metal, CUDA, SYCL, OpenCL, WebGPU, RPC). The
following are byte-unchanged: `ggml-cpu/kleidiai/`, ARM feature detection
(`arch/arm/cpu-feats.cpp`), `ggml-cpu/CMakeLists.txt`, `ggml/src/CMakeLists.txt`
and `ggml/cmake`. KleidiAI is still v1.24.0 and its `kai` tree digest is
unchanged. The `ggml/CMakeLists.txt` delta is the ggml version bump
(0.24.0 to 0.25.1).

Only one change touches ARM code: 8034c1d1f adds Q1_0 repack kernels
(`ggml_gemv/gemm_q1_0_4x{4,8}_q8_0`). Each NEON body is compile-time guarded
(`__ARM_FEATURE_DOTPROD` for the 4x4 kernels and the 4x8 GEMV,
`__ARM_FEATURE_MATMUL_INT8` for the 4x8 GEMM) and falls through to the
existing generic C implementation. `arch-fallback.h` maps the new names on
non-ARM targets. Runtime selection in `repack.cpp` follows the existing
repack pattern. It requires `ggml_cpu_has_neon()` plus
`ggml_cpu_has_matmul_int8()` or `ggml_cpu_has_dotprod()`, and `ne[1] % 4 == 0`.
A variant compiled without a feature therefore runs the generic body, even
when runtime detection selects the kernel. The kernels contain no SVE or SME
code.

The other CPU changes are ISA-neutral. They add F16 `src1` to the
Hadamard/FWHT path of `ggml-cpu.c`/`.cpp`/`ops.cpp` (using the existing
`ggml_cpu_fp16_to_fp32`), add hc ops, and fix a SpacemiT RISC-V transpose.
Shared changes are: a scheduler graph-reserve failure check (#26070), meta
backend buffer-view resolution (#29266), the hc op declarations in `ggml.h`,
a `ggml_permute` stride-truncation fix (#29227), IQ1_M reference quantization
building prefix sums once per block (#28706), and GGUF data-section alignment
relative to the GGUF start (#28993). The added lines of the shared and
generic CPU files contain no intrinsics, inline assembly or new `#if` guards.
No change raises the baseline ARM ISA or alters KleidiAI kernel selection,
packing or its callers.

The accepted pair binds the full trees, including the accelerator changes:

- ggml/src: `68d5bc369749e78545f50dd5107368ec7ee1874619794795cd142b0043c747f6`
- kai: `64189fc613c1c4c3aaeeb6bb12b38d85dd6728cafd2261a5a88f1b77b10fe59c`

Unknown source combinations, and scalable instructions outside the existing
18 exact ELF function ranges, remain rejected.

## Artifact and execution evidence

The exact upstream SHA was built locally with the release helper's
`android_armv8.2_2` CPU variant and NDK `28.2.13676358`. Before the pair was
added, the production validator failed with the identical fingerprint pair
reported by the release run. With the pair added, the unmodified instruction
containment policy inspected 260,601 instructions. All 2,187 scalable
instructions, the same count as v0.4.1, were contained in the existing 18
functions. In this non-I8MM variant, `ggml_gemm_q1_0_4x8_q8_0` compiles to a
single tail branch into the generic kernel and contains no `smmla`. The other
three Q1_0 kernels use `sdot`, which the variant's `GGML_USE_DOTPROD` requires.
The local `libggml-cpu.so` SHA-256 was
`20543a590f8f4e92ff0087cf364e43f785331006ddabe27b1f9eb7f16e694d70` (macOS
NDK host). This CPU artifact check does not claim full Vulkan/OpenCL packaging
or hardware execution coverage.

`Validate Wrapper` keeps the prior candidate lanes and adds exact v0.5.0 rows:

- Android release ARMv8.2 CPU artifact build and source/instruction validation.
- Compiled selector and Q4/Q8 scalar-reference compute under QEMU `cortex-a53`
  and `max,sve=off,sme=off`, both normally and with guest
  `GGML_KLEIDIAI_SME=1` to exercise the unsupported-feature fallback.

Hosted results and an independent exact-head review must pass before merge and
be linked from the PR. Physical Android and SME hardware execution is not
performed here (N/A), and emulator evidence is separate from device
qualification. Release publication and consumer adoption need their own
approval.
