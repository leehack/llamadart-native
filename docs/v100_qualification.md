# v1.0.0 native candidate

Owner: [native #112](https://github.com/leehack/llamadart-native/issues/112),
coordinated by [core #2024](https://github.com/leehack/llamadart/issues/2024).
This is an experimental native-source candidate, not a qualified release.
Proposed artifact version **v1.0.0** follows
[the stable-version policy](release_version_policy.md), independently of core
0.20.0. Merge, preparation, hardware/model execution, publication and consumer
adoption have separate gates.

| Input | Exact revision |
| --- | --- |
| Owner base | `b99859446a8349ac9eb93b97dcc9908d28cfb356` |
| Published native v0.9.0 owner | `a604639131105c42a25a3e7384ca10314d385f80` |
| Baseline llama.cpp | `cd959bae99ff4bed22bef2604f45241c09bbb8c8` |
| Fixed candidate | `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` |
| Stable annotated tag object | `a6d1a74dbebce3a16f4f2ceddbf38c21a08d8674` |

The dereferenced published `v1.0.0` tag supplies the candidate. The release
object's `target_commitish=2d8a3250…` is a different commit and is not used.
No nightly/main changes, new speech admissions, cache types or public controls
are part of this candidate. Both v0.8.0 and v0.9.0 validation rows remain.

## Carry decisions

No new retirement is proven. The required removal evidence remains original
trigger **old unpatched FAIL / old patched PASS / candidate unpatched PASS**, with
other fixes held constant. Application, marker presence, issue closure and a
successful build cannot replace that evidence. Patch bytes and their digests
remain unchanged; [series.json](../patches/llama.cpp/series.json) is authoritative.
0008 and 0009 are the only additions, below.

| Carry and origin | Original trigger and upstream status | Candidate decision and proof/gap |
| --- | --- | --- |
| 0001, [native #79](https://github.com/leehack/llamadart-native/issues/79), introduced by #104 / `c766a94a4d48` | SC-51E/API36, Adreno 750, Qualcomm proprietary driver `2150604839`, build `c81b28e1a5`; stories15M F32 / Qwen3.5-0.8B Q4_0 multi-token prefill crashes in `vkCreateComputePipelines`. [Upstream #29165](https://github.com/ggml-org/llama.cpp/pull/29165) remains an unmerged draft. | Retain exact Android driver match and shared-memory reduction. Strict application passes. Original old/patched receipts are in #79; v1 same-driver execution is pending. F32 fusion and quantized numerical defects survive the crash repair: retain consumer GPU refusal. No global fusion-disable prototype is added. |
| 0002, [core #782](https://github.com/leehack/llamadart/issues/782), #104 / `c766a94a4d48` | SC-53C/API36 Mali-G68, feature true with null `vkGetBufferDeviceAddress`; original report is **stable-diffusion.cpp**, with llama.cpp exposure unverified and original device API unrecorded. v1 has no equivalent entry guard. | Retain null-entry guard. Strict application passes; no native affected-device patched PASS is claimed. Need null/present-entry causal controls and actual native reachability/API before any retirement. Below-1.2 consumer refusal is separate. |
| 0004, [native #154](https://github.com/leehack/llamadart-native/issues/154), #153 / v0.8.0-1 | Old macOS/iOS bfloat Metal compiler rejects unary `constexpr` vector subscripts. [#31360](https://github.com/ggml-org/llama.cpp/pull/31360), `c9daee149d606adc30b4c5b7a3254697c9771e2b`, is integrated. | Retain patch for v0.8 validation. Existing `upstream_fix` path already skips v0.9 and v1; this PR does not retire it. [Old red/green receipts](v080_1_wrapper_rebuild.md) remain controls; exact v1 unpatched old-OS/bfloat compiler and physical iOS proof is missing. |
| 0005, [native #174](https://github.com/leehack/llamadart-native/issues/174), #177/#182 | Non-SVE NEON FP16 dot overflow at magnitudes 80/90/100; macOS arm64 and optimized Android/Linux CPU modules. [#31734](https://github.com/ggml-org/llama.cpp/issues/31734) closed after [#31854](https://github.com/ggml-org/llama.cpp/pull/31854), `12ab5979869aecdf3593de260a99ca91263bf5a1`, changed default graph routing. | Retain. v1 still contains the old F16 kernel. Model-free direct-kernel controls fail when only 0005 is removed, while the new graph route passes. Keep exact blocked/flush/tail/nonfinite/F32-recompute controls and old rows; no retirement. |
| 0006, [native #174](https://github.com/leehack/llamadart-native/issues/174), #179 | FP16 llamafile GEMM, >=8 columns, macOS arm64 / Linux armv8.2_2. Android has llamafile off; SVE/I8MM compile it out. Same upstream graph change leaves F16 x F16 tinyBLAS present. | Retain. Removing only 0006 fails direct GEMM overflow/512-K block/tail/last-cell controls while v1 graph passes. The narrow #174 policy exception is unchanged. Actual module/model throughput remains pending. |
| 0008, [native #192](https://github.com/leehack/llamadart-native/issues/192) | v1 runs F16 `src0` matmuls on `ggml_vec_dot_f16_f32` ([#31854](https://github.com/ggml-org/llama.cpp/pull/31854), `12ab5979`): on ARM FP16 modules 0.16 to 0.6 of v0.9.0 for F16 weights (Neoverse-N1), with 0005/0006 unreachable from graphs. | Add. An unhinted F16 x F32 `GGML_OP_MUL_MAT` takes the v0.9.0 route (query rounded to F16) on FP16 vector arithmetic modules; F32/BF16/Q8/Q4 `src1` requests and `CONV_2D`/`CONV_3D` stay F32. Maintainer-approved exception; see [native192_arm_fp16_f16_route.md](native192_arm_fp16_f16_route.md). Neoverse-N1 (t2a) 0.99–1.07 outside FA-on depth decode (0.95–0.97; the one other cell below 0.97, armv82 Q8 tg128 FA-on 0.967, is within the identical-module null band); Android gate-1 rerun at #193 head: CPU ids identical to v0.9.0, C1 passes. x86 AVX2 (#188 candidate): 20 cells, worst 0.978. V2 and c4a pending. |
| 0009, native #192 | Same change for the flash-attention one-chunk K.Q dot. | Add, same route for an F16 K. Removing it fails the `flash-route` row. |

0003 was deleted by #185 when the v0.7 row went; it is not restored. See
[native174_f16_dot_accumulation.md](native174_f16_dot_accumulation.md) for the
original numerical contract, historical provider/device receipts and accepted
performance costs. Those receipts are not candidate qualification.

## Integration protections

All eighteen audit integration protections remain. Local CPU contract tests are
component evidence; the last column names the remaining original environment.

| ID, origin | Retained behavior / candidate adaptation | Remaining qualification |
| --- | --- | --- |
| I01, [#15](https://github.com/leehack/llamadart-native/pull/15), core #169 | Mobile Apple constructor `GGML_METAL_NO_RESIDENCY=1`, overwrite=0; process-wide/load-order semantics unchanged. | iPhone13/iOS18.3.1 Qwen2-0.5B memory-pressure/compiler trigger, env/override controls; no retirement from VM-allocation or unary changes. |
| I02, core #613, native #99/#100 | Apple stage-ordered exit tracking, final-free admission and guard through synchronization. | stories15M Metal exit134 and in-flight final-free replay on actual GPU. |
| I03, native #105/#109, core #977 | Linux owned-static retention, non-IPO shim and NODELETE; callbacks, stdio and cooperative joins preserved. | GCC/HIP ownership and actual GPU/BLAS cooperative shutdown. Raw exit with live Dart isolates stays unsupported. |
| I04, native #103/#102 | Exception/result ownership, Windows noinline classification and `/EHsc-`, Android single C++ runtime ownership unchanged. | Real Windows compiler and Android cross-DSO exception/import tests; discard failed Windows objects. |
| I05, native #74/#127/#145 | Isolated ARM variants, required-feature scoring, unscored Linux baseline, exact source-pair/range policy. | Candidate compiled dispatch, Android ELF ISA, QEMU/non-SVE compute and separate ggml SVE256 controls. See [the source/ISA record](v100_android_isa_qualification.md). |
| I06, native #74 | Windows ARMASM preprocessing and restored exact ClangCL kernels with source-local flags. | VS18/ClangCL build, full mask selectors and baseline ISA containment. |
| I07, `aced14990275`, native #39 | Per-target MSVC common/mtmd IPO suppression retained. | Same-toolchain fail/pass/candidate proof before reversing defaults. |
| I08, upstream #29373 | NDK28.2, Ubuntu24.04 host glslc override and runtime coopmat feature gate retained. | Exact release shader compiler lane and actual extension/opt-out controls. |
| I09, shipped presets | Consolidated Apple Metal+CPU and embedded sources; BLAS/Kleidi/BF16 defaults unchanged. | Actual Apple OS-family/BF16/old-OS compiler evidence. CPU success is not Metal qualification. |
| I10, native #72/#149 and v0.6 adapter | Compile-only C++17 probes; synchronous copied callback storage, section-major M-RoPE, metadata/status. Adapt relocated `mtmd-impl.h` projector parser, preserving 70..1120 Gemma image-cap refusal. | All retained rows, wrong-probe controls and real media callback/cancellation. |
| I11, native #113/#115 | Zero draft MoE cache, preserve target budget and independent batch/microbatch; new CPU-share default stays 0. | Actual same/external draft/cache model controls; no new public tuning. |
| I12, native #157/#40/#51 | Fresh-target check over all sequences, shared-KV probe ordering, synchronize-before-clear, FULL/NO refusal, block output sizing. | Genuine MTP/DFlash/DSpark pair and FA controls; generated CPU components do not qualify trained generation. |
| I13, core #1403, native #158 | Creation-time ownership, never-reused tokens, cross-owner refusal, exactly-once stage draining. | Consumer kill-at-return with actual candidate module hash before/after hooks. |
| I14, native #87/#141/#117 | Guard-through-synchronize, original exception priority, atomic cancellation and safe audio/media graph splits. | Actual backend PCM/output parity and latency; inspect changed upstream fusion separately. |
| I15, native #103 | Backend/CPU sampler accept-once, EOG stop and mismatch semantics retained. | Actual pair/ngram reference and backend-selected token controls. |
| I16, native #122/#180/#176 | Existing generation RNG and suppress-token behavior retained. Explicit frame-temperature adapter preserves LFM3 model default and v0.9 Pocket pack policy, separately from talker temperature. | Actual same-seed/reset/cancel/reuse PCM of admitted speech families. No DRAMA3/live-audio expansion. |
| I17, native #128/#54 | Scalar build identity, state compatibility refusal, 94 required / 15 retired wrapper exports and package/dependency ownership. Compiled layout/default canaries added. | Published archive/header/manifest provenance, every ABI/module and actual restored-state output; consumer pre-struct admission remains downstream work. |
| I18, core #996/#948 | Consumer API>=1.2 and exact-driver Vulkan refusal; missing/old-runtime Qwen3.5 and projector CPU restrictions preserved by leaving consumers unchanged. | Actual placement/refusal and original same-driver numerical evidence before any separate admission change. |

## ABI and used-export handoff

`llamadart_upstream_abi_test` checks defaults only after scalar build-identity
admission. Its negative old/missing/unknown/dirty/duplicate-source controls must
make **zero** by-value calls. This is an owner test/handoff harness: the later
consumer must implement admission before its first by-value function, including
speculative capability/default probes and framework overrides. The existing
`llama_dart_build_identity()` is object-free and sufficient to identify the exact
upstream; no new public probe/export is necessary. A source allowlist alone does
not establish packaging provenance: verify the resolved library/framework and
published owner/header hashes as well.

Compiled macOS arm64 canaries (other shipped ABIs must run PR/platform lanes):

| Contract | v0.9.0 | v1.0.0 |
| --- | --- | --- |
| `llama_context_params` sizeof / alignof | 168 / 8 | 176 / 8 |
| `moe_cache_size` / `abort_callback` offsets | 112 / 120 | 112 / 128 |
| `samplers` / `ctx_other` offsets | 144 / 160 | 152 / 168 |
| `moe_cache_cpu_share` | absent | offset 120, default 0 |
| `llama_model_params` size | 80 | 80; `load_mtp` remains default false |
| `mtmd_helper_gen_audio_inp` size | 64 | 72; temp offset 56, seed 60, upstream temp default -1 |
| `GGML_OP_COUNT` | 102 | 99 |
| Optional RPC wire version | 8.0.1 | 10.0.0; match peers before use |

The complete used C-signature/export inventory includes maintained owner calls,
consumer calls, lazy symbol strings and function tear-offs, excluding generated
binding declarations. At published core `7b8f5e57b817abf6414b3453d4fa13ed52cf83f2`,
328 distinct referenced public symbols were inspected using Clang FunctionDecl
signatures and the built consolidated library's exports. NextN declarations
promoted from `llama-ext.h` keep their used signatures. All referenced candidate
symbols are exported **except removed `llama_token_get_text`**, called by core's
`_promptStartsWithBosToken` (`llama_cpp_service.dart:6216`); the consumer lane must
replace it with `llama_vocab_get_text` and regenerate bindings. Do not restore the
removed export to hide this migration. Wrapper common/speculative/sampling/chat
interfaces were separately diffed and compiled; no used signature adaptation was
needed. The `llama_set_embd_state_src` spelling was introduced after v0.9 and
renamed to `llama_set_ctx_state_src` before v1; neither is called by maintained
owner/core code in this inventory. It is not a removed v0.9 call.

## Audio and numerical controls

New `temp=0` means greedy/zero noise. The wrapper selects `temp=-1` for LFM3 and
untuned variants; Pocket `english` / `english_2026-04` explicitly retain **0.3**,
`english_2026-01` / `french_24l` retain **0.7**, matching v0.9's removed helper
pack overrides. Missing/unknown/Italian variants retain the model default. Old
headers keep the previous zero-initialized field set without a temp member.
The independent field check rejects a false temperature probe. This preserves
the input policy, not a claim of same PCM across changed upstream models.
Talker request temperature, model suppress tokens and per-generation seed stay
separate. Chatterbox's helper still identifies Multilingual by position tables;
the owner keeps the corresponding capability/paired-sequence refusal.

Graph tests require the route expected by the retained upstream row and the
module: v1 without 0008 and 0009 uses F16 weights with unrounded F32 queries;
with them, FP16 vector arithmetic modules round an unhinted query to F16 as
v0.8/v0.9 do and use the exact F16 blocked oracle, and the `route-*`,
`conv-route` and `flash-route` rows pin the route. The precision-setter adapter uses
`ggml_prec_set_acc` when declared, with the old setter fallback retained.
`--residual` calls the **same compiled CPU library's** old F16 dot and tinyBLAS
kernels, preserving bit-exact dot/F32-recompute, tail, flush, nonfinite,
512-K GEMM and partial-tile controls. An unavailable non-FP16/SVE residual path
is explicitly skipped, not qualified. A missing kernel symbol on an affected
module fails. The direct GEMM uses the backend's initialized one-thread pool.
Removing 0005 or 0006 separately fails on v0.9 and v1; on v1 the graph still
passes. Candidate direct-kernel failure rules out retirement from graph bypass.

[CPU-FA #187](https://github.com/leehack/llamadart-native/issues/187) / upstream
#29810 is a separate V-accumulator fix. Pending reproduction uses closest
Nemotron3-Nano4B Q8_0, ~8196-token context and short final batch, FA on/off,
F32-V and short-context controls, finite/logit/output bounds and real x86/ARM
module performance. No substituted-model proof is claimed. Upstream #30541
fixes quantized ON_DEVICE state copying: require q8_0/q4_0 save/restore/scatter,
fresh-vs-restored output with FA on and f16/f32 controls, recording actual flags,
module and backend. Ordinary serialized-state fixture roundtrips are not proof
that ON_DEVICE executed.

## Execution and remaining gates

Bulky logs, source/header inventories, compiled-source mutation receipts and
metrics are under
`/opt/UnitySrc/personal/llama/llamadart-evidence/planning-next-release/owner-candidates/llamadart-native/implementation/`.
The sibling read-only audit/report/matrix and earlier llama-cpp plan are source
inputs, not executed candidate evidence. Each execution receipt must name owner
head, upstream, patch/source digests, compiler, ISA and actual backend; later
model/device receipts additionally need exact archive/module/model/projector
hashes, device/OS/driver and settings.

Local implementation checks (2026-11-15): v0.8/v0.9/v1 each pass **88/88**
native CPU fixture tests; **449** Python tests pass with one Linux-only host
skip on macOS. Strict series application passes for all three rows. The candidate
passes 94 required wrapper exports, retired-export absence, no `__cxa_atexit`
imports, build identity and release-provenance contracts. Workflow lint passes
with the installed linter's existing VS2026 runner-label exception. The candidate
Android module and native Kleidi selector/compute evidence are in the linked ISA
record. Zero audio initialization fails the new temperature test; a blind -1
Pocket default also fails. Wrong/missing temperature probe and old context size
fail compilation; bypassing scalar admission fails the no-by-value-call spy.
These are component controls, not platform/model qualification.

Throughput of F16 weights and F16 KV on ARM FP16 modules against v0.9.0 with
0005/0006: without 0008 and 0009 M4 Max F16 pp512 0.25–0.28 and Neoverse-N1
0.16; with them M4 Max 0.98–1.01 and Neoverse-N1 matmul-bound cells 0.99–1.07 (details and what is still pending in
[native174_f16_dot_accumulation.md](native174_f16_dot_accumulation.md)).

Local checks use CPU-only native fixtures generated by the tests, never
pretrained models. Configure without `LLAMADART_EXIT_TEARDOWN_TEST_MODEL` or
`MMPROJ`; leave TTS/embedding smoke executables off. Inspect CTest commands first:

```sh
cmake -S . -B <evidence>/build-v1-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMADART_BUILD_TESTS=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build <evidence>/build-v1-cpu -j 6
ctest --test-dir <evidence>/build-v1-cpu -N -V
ctest --test-dir <evidence>/build-v1-cpu --timeout 45 --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/llama_cpp_patches.py apply --strict --upstream third_party/llama.cpp --output <evidence>/strict-v1
python3 tools/validate_exports.py --format nm --tool nm --forbid-import __cxa_atexit <evidence>/build-v1-cpu/libllamadart.dylib
python3 scripts/verify_release_provenance.py
```

Before readiness: fresh independent exact-head/current-base audit; all ordinary
PR lanes including Windows/Android compiled dispatch and ISA, Linux variants /
glibc / package / host shutdown, Apple OS-floor compiler lanes and all ABI rows.
OS-floor CI does not cover macOS13/14 or iOS16/17 physical hardware. Before
publication/adoption: separately approve nonpublishing release preparation,
qualify its exact bytes on original S24/A53/old Apple triggers and affected CPU
modules, existing text/speculation/media/embedding/speech/model lifecycle packs,
CPU-FA and ON_DEVICE state controls. No manual workflow/device/GPU/model run,
release, tag or consumer pin/binding/checksum change is authorized by this PR.
