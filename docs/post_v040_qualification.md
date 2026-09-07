# Post-v0.4.0 correctness qualification

Issue #76 qualifies exact upstream commit
`73ab7599b553c03f6f5d2db24a18ad76f2eb36a3`, separately from the production
submodule and published v0.4.0. It contains the grammar repetition threshold
fix (#28469), GDN normalization (#28068), Metal early-return leak (#28399),
and CUDA race/barrier fixes (#28475/#27870).

`validate_wrapper.yml` retains production-pinned Windows ARM64, Linux wrapper,
Linux x64/arm64 artifact, and Windows x64 link checks. Its explicitly named
post-v0.4.0 counterparts fetch and verify the immutable candidate SHA.
Android ISA and emulated KleidiAI qualification also target this candidate.
The jobs have read-only repository permissions and do not dispatch, publish,
tag, ingest attestations, or change any release pin. Candidate failures block
candidate readiness, not an assertion that currently pinned artifacts regressed.

The candidate Linux wrapper additionally calls the built library's public
grammar sampler through `tools/validate_grammar_boundary.py`: 1999 and 2000
repetitions must initialize, 2001 and malformed syntax must fail. The unchanged
v0.4.0 library fails this test at 2000, so the check runs only on the candidate.
Character-only initialization requires no model vocabulary; this is sampler
initialization evidence, not model inference or token acceptance coverage.

Local macOS CPU/Metal wrapper tests and exports, a bounded Qwen3.5 model A/B,
and matching bridge state/image/ASR/TTS smokes are recorded in issue #76.
They do not replace the hosted owner-platform checks or the hosted exact-artifact
Web qualification required for publication. CUDA hardware and older Android
remain unavailable/unverified; emulation is not a physical-device result.

This qualification does not invent a release identity: later upstream source
must not be called a v0.4.0 wrapper rebuild. Adoption requires a containing
upstream stable release or a separately authorized development-channel decision.

## Android ISA source audit

The first candidate build succeeded but correctly failed the pre-existing source
fingerprint gate. A separate audit compared v0.4.0 against the exact candidate:
`ggml/src/ggml-cpu`, CPU feature detection, KleidiAI selectors/tables/callers,
`ggml/include`, and ggml build configuration are byte-unchanged. Of 37 changed
files under `ggml/src`, 36 concern accelerator backends. The shared
`ggml-backend.cpp` delta changes two log levels and removes a capacity-induced
scheduler split; it introduces no optimized-kernel call or feature-mask bypass.
The candidate's compiled selector/quantized-compute test passed under non-SVE
QEMU. Independent Astra source-delta review confirmed this evidence.

After explicit maintainer approval, the containment policy retains the existing
v0.4.0/KleidiAI pair and adds only candidate ggml SHA-256
`dcb0f04ebb9654b1fe5ac7cc45737c79e62b116a2063ceda81a7ec1ddb1b20e2`
paired with unchanged Kai SHA-256
`64189fc613c1c4c3aaeeb6bb12b38d85dd6728cafd2261a5a88f1b77b10fe59c`.
Unknown or cross-combined pairs fail closed. Exact ELF function-range allowlists
and scalable-instruction checks are unchanged. Replacement hosted Android
disassembly remains required; accepting source identity alone is not a pass.
