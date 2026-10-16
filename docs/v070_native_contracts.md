# v0.7.0 native contracts and EmbeddingGemma-2 qualification

Qualification candidate: llama.cpp `v0.7.0`,
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. Production base remains
`d81235049384534c167caea52b85a694f6103d14` (`v0.6.0`, currently distributed
as native `v0.6.0-3`). This work does not change the production submodule,
publish artifacts or update the downstream pin.

Trackers: [native #112](https://github.com/leehack/llamadart-native/issues/112),
[native #113](https://github.com/leehack/llamadart-native/issues/113),
[native #114](https://github.com/leehack/llamadart-native/issues/114),
[Dart #1072](https://github.com/leehack/llamadart/issues/1072),
[Dart #1123](https://github.com/leehack/llamadart/issues/1123).
The Android ISA source-pair audit and hosted candidate lanes are described in
[v070_android_isa_qualification.md](v070_android_isa_qualification.md).

## Draft expert cache

v0.7.0 introduces `llama_context_params.moe_cache_size`. Upstream's
`common_speculative_context_params` clears it for draft contexts. Our generic
speculative and legacy MTP paths construct contexts directly, so both must
apply the same restriction before `llama_init_from_model`. A declaration probe
keeps the helper compatible with older headers. The guard changes only the
draft copy; target parameters and draft `n_batch`/`n_ubatch` remain independent.
Contract assertions cover a nonzero target budget, zero draft budget,
idempotence and preserved batch sizes, with the older-header no-op compiled
in the v0.6.0 lane.

This does not expose a new public configuration option. The downstream cache
budget and draft microbatch controls have separate trackers.

## Exports and ABI handoff

ELF validation now reads only `.dynsym` (`--dyn-syms --wide`) and accepts defined
GLOBAL/WEAK symbols with DEFAULT/PROTECTED visibility. A name found only in
`.symtab`, or a LOCAL/HIDDEN/INTERNAL definition, cannot satisfy the export gate.
Negative CLI tests cover private and static-only definitions. Import checks
also use dynamic symbols. Mach-O and PE inspection remain unchanged.

v0.7.0 inserts `size_t moe_cache_size` between `type_v` and `abort_callback` in
`llama_context_params` and adds counters to `llama_perf_context_data`. Therefore
old generated bindings cannot safely consume the candidate's by-value context
parameters. Regenerate from the exact published header bundle and validate
native/Dart size and offset agreement before downstream adoption. The smoke
prints native context size, relevant offsets and performance-struct size for
that handoff. Upstream scheduler callback exports also need bundle-level
qualification; the existing wrapper export list alone does not establish
complete downstream API coverage.

## Dynamic embedding extraction repair

The original v0.7.0 candidate fails the existing `model-session` test with a
CPU segfault. A minimal program using only upstream C calls also crashes:
create a default context, enable embeddings, decode 32 tokens. The same
program built with v0.6.0 headers and library exits successfully. LLDB places
the fault in `ggml_gallocr_alloc_graph`, reached through the scheduler and
`llama_context::decode`; no wrapper or speculative function is needed.

The carried `0003-embedding-mode-scheduler-reserve.patch` marks the scheduler
for reservation whenever embedding extraction changes, including pooling
NONE. The upstream conditional only handles pooled extraction. This repairs
the demonstrated allocator-assignment reuse failure without changing the
unchanged-mode path. A debug log provides the mandatory Android bundle marker
in `libllama.so`. It fits both v0.6.0 and v0.7.0; the production submodule stays
untouched. An upstream-equivalent repair in a qualified release removes this
patch. The precise upstream introducing commit remains unproven.

The durable `model-embedding-toggle` test uses direct upstream calls with no
speculative state, enables and disables extraction twice, checks output
availability and frees the context/model. The existing session test retains
its independent speculative teardown coverage. Do not skip either test to
qualify a candidate. This is a candidate regression repair, not evidence that
the currently published native v0.6.0-3 artifact has this crash.

## EmbeddingGemma-2 model smoke

`tools/embedding_smoke.cpp` is an opt-in CPU qualification executable. It:

- Checks input width 512 and pooled output width 768, finite nonzero values,
  and truncate-then-normalize handling for 128/256/512/768 dimensions.
- Compares the guarded wrapper's text decode against direct upstream decode.
- Encodes synthetic images and 16 kHz audio with the guarded mtmd calls, then
  constructs one complete mixed token/embedding batch through upstream's
  extended-batch API. All noncausal inputs must fit one microbatch.
- Rejects silently discarded media and checks that different image/audio
  content changes the output. A later text request must match the original.
- Frees tracked models/contexts and checks that the tracking registry is empty.

The mixed processing call is an **upstream reference path**, not a new guarded
wrapper export or public Dart media-embedding API. Do not reuse chat's
chunk-by-chunk decode helpers for noncausal mean pooling: separate forwards do
not preserve full-input attention and pooling. Native guarded mixed processing,
capability/version-skew handling, request limits and cancellation remain part
of #112/#1072. Synthetic content sensitivity is a wiring check, not a retrieval
quality benchmark or image/audio/text semantic alignment evaluation.

### Exact assets

Public [ggml-org/embeddinggemma-2-GGUF](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF),
revision `bfcd298762cc34d0357ece5ebdd31791a3a374d8`:

| File | SHA-256 |
| --- | --- |
| `embeddinggemma-2-BF16.gguf` | `68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216` |
| `mmproj-embeddinggemma-2-BF16.gguf` | `d2033b3cd0223cfb2e7a2b50dc80a664c78c816d375f5b017ec3ef80ee0bc766` |

Use matching BF16 artifacts. The [model card](https://huggingface.co/google/embeddinggemma-2)
documents task prefixes, output truncation and FP16 precision limitations.
No weights are downloaded by default tests or committed to this repository.

### Reproduce

In a disposable qualification checkout, select the exact candidate above and
initialize submodules. Supply verified local asset paths:

```bash
cmake -S . -B build/embedding-v070 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLLAMADART_BUILD_TESTS=ON \
  -DLLAMADART_BUILD_EMBEDDING_SMOKE=ON \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_METAL=OFF -DGGML_NATIVE=OFF -DGGML_CCACHE=OFF \
  -DLLAMADART_EMBEDDING_TEST_MODEL=/path/to/embeddinggemma-2-BF16.gguf \
  -DLLAMADART_EMBEDDING_TEST_MMPROJ=/path/to/mmproj-embeddinggemma-2-BF16.gguf
cmake --build build/embedding-v070 --target llamadart_embedding_smoke -j 8
ctest --test-dir build/embedding-v070 -R '^llamadart_embedding_smoke$' --output-on-failure
```

On other platforms, explicitly disable their GPU backends as well to reproduce
the CPU row. Default CPU dispatch and other backend lanes remain independent.
Restore the production submodule before committing this qualification work.

### Local evidence

2026-10-16, Apple M4 Max, macOS arm64, CPU/Accelerate, KleidiAI and Metal off,
eight threads, context/batch/microbatch 2048, exact candidate and BF16 assets:

| Check | Result |
| --- | --- |
| Python suite, including patch markers and export negatives | PASS; 213 tests, one platform skip |
| v0.6.0 native contracts with the carried repair | PASS; 48/48 |
| v0.7.0 native contracts and opt-in embedding smoke with repair | PASS; 49/49 |
| Wrapper vs upstream text, normalized 768-vector | PASS; maximum component difference 0 |
| Black vs checkerboard image, mixed text/media reference | PASS; maximum component difference 0.074377097 |
| Silence vs 440 Hz tone, mixed text/media reference | PASS; maximum component difference 0.069904178 |
| 128/256/512/768 prefixes finite and normalized | PASS |
| Text reuse after image/audio requests; tracked cleanup | PASS |

Native arm64 ABI report: `sizeof(llama_context_params)=168`, `type_v` offset
108, `moe_cache_size` offset 112, `abort_callback` offset 120;
`sizeof(llama_perf_context_data)=56`. These are measured native values for this
build, not portable constants to copy into generated bindings.

This does not qualify Metal, Vulkan, CUDA, mobile devices, Web, FP16 or
quantized variants. Hosted candidate/production lanes, published artifacts,
retained-patch qualification, refreshed Dart bindings and consumer model-backed
smokes remain gates before adopting v0.7.0.
