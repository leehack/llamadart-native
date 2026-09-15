# v0.4.1 Android CPU ISA qualification

The Android Vulkan and OpenCL builds in native release run `34948903228`
compiled successfully, then rejected the new complete `ggml/src` fingerprint.
Both jobs selected upstream v0.4.1 at
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`. This policy update does not change
the production submodule pin, release channel, or dispatched function allowlist.

## Source review

Compared with the previously audited `73ab7599b553c03f6f5d2db24a18ad76f2eb36a3`,
129 files under `ggml/src` changed: seven CPU, five shared, and 117 accelerator
files. ARM architecture sources, KleidiAI integration, feature definitions,
backend registry, and `ggml/src/CMakeLists.txt` are unchanged. CPU changes cover
s390 support, a VXE repack path whose feature probe returns false on ARM,
a ClangCL vector initializer correction, and cache-line sizing. Shared changes
include empty scheduler ID tensors, precision API declarations, and errno use.
No change raises the baseline ARM ISA or alters ARM kernel selection.

Runtime feature detection still precedes SVE vector-length queries. Kernel
selection requires the complete feature mask and supported vector length.
`GGML_KLEIDIAI_SME=1` cannot create absent hardware capabilities; fallback
removes SME/SME2 before selecting a supported kernel. Packing and compute call
sites retain these selectors. Independent source review found no blocker.

The accepted pair binds the full trees, including accelerator changes:

- ggml/src: `bf7ae6d2ea861ce6cd4b56afce154a45461df79b46c02e340d1adfe0075467b5`
- kai: `64189fc613c1c4c3aaeeb6bb12b38d85dd6728cafd2261a5a88f1b77b10fe59c`

Unknown source combinations and instructions outside the existing 18 exact
ELF function ranges remain rejected.

## Artifact and execution evidence

The exact upstream SHA was built locally using the release helper's
`android_armv8.2_2` CPU variant and NDK `28.2.13676358`. The unmodified
instruction containment policy inspected 256,196 instructions; all 2,187
scalable instructions were contained in the existing 18 functions.
The resulting `libggml-cpu.so` SHA-256 was
`83cc18799d67287f2f1353d3d58c03e52f421a680dd5c7069411f0bea23a06f7`.
This CPU artifact check does not claim full Vulkan/OpenCL packaging or hardware
execution coverage.

`Validate Wrapper` retains prior candidate lanes and adds exact v0.4.1 lanes:

- Android release ARMv8.2 CPU artifact build and source/instruction validation.
- Compiled selector and Q4/Q8 scalar-reference compute under QEMU `cortex-a53`
  and `max,sve=off,sme=off`, both normally and with guest
  `GGML_KLEIDIAI_SME=1` to exercise unsupported-feature fallback.
- Windows ARM64 release preset, which explicitly uses ClangCL, and native
  wrapper contracts to cover the compiler-specific initializer change.

Hosted results and independent exact-head review must pass before merge and
be linked from the PR. Physical Android and SME hardware execution are not
performed here (N/A); emulator evidence is separate from device qualification.
Release publication and consumer adoption require their own approval.
