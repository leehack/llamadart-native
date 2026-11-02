# Generic MTP initialization and rollback

[Native #51](https://github.com/leehack/llamadart-native/issues/51) tracks the
remaining generic speculative migration. This change restores the legacy MTP
rollback check in `llama_dart_speculative_init`; it removes no ABI exports and
does not retire the legacy entry points.

For `draft-mtp`, the caller must reset **every sequence** in target memory
before initialization, then ingest the prompt afterward. Initialization rejects
a populated target before creating a draft context or changing target state.
The restriction also applies when MTP is combined with a draftless strategy.
N-gram and ordinary draft-model initialization retain their existing behavior
on populated targets.

The rollback probe decodes two dummy target tokens, then two draft tokens
while the target KV remains available. Shared-KV drafts such as Gemma4Assistant
read target KV; probing them after clearing the target can abort non-flash
attention on an all-masked softmax. Only after both decodes does the probe
attempt partial removal, then synchronize and clear both contexts on every
return path. A fresh-memory precondition protects caller state. Checking only
sequence zero would discard other populated sequences. `llama_memory_can_shift`
does not establish rollback support, and an unrelated scratch context does not
establish the supplied target's configuration.

Once a fresh target and MTP draft exist, both must report partial removal
(`PART`) or recurrent snapshots (`RS`). Full-only removal (`FULL`), no memory
or failed decode (`NO`) rejects initialization, releasing the draft. The
existing C ABI failure result is `NULL`, with a diagnostic. Ordinary
speculative strategies can use different rollback mechanisms and are not
subject to this MTP check.

## Local validation

Use the pinned submodule and a Debug build, with assertions enabled:

```sh
cmake -S . -B build/mtp-rollback -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug -DGGML_CCACHE=OFF -DGGML_OPENMP=OFF \
  -DGGML_METAL=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/mtp-rollback --target \
  llamadart_speculative_api_test llamadart_exit_teardown_test
ctest --test-dir build/mtp-rollback -R 'llamadart_speculative' \
  --output-on-failure
build/mtp-rollback/llamadart_exit_teardown_test \
  make-vocabulary-model /tmp/speculative-target.gguf
build/mtp-rollback/llamadart_exit_teardown_test \
  make-mtp-model /tmp/speculative-mtp.gguf
build/mtp-rollback/llamadart_speculative_api_test \
  /tmp/speculative-target.gguf /tmp/speculative-mtp.gguf
```

The generated dense MTP-head component has matching dimensions and vocabulary
with the generated decoder. It exercises actual context creation and native
rollback probes; its weights are not trained. The test requires byte-identical
target state after rejected initialization with either sequence zero or only
sequence one populated, successful n-gram and ordinary draft initialization
with those populated targets, successful fresh MTP initialization, independent
target/draft probe-decode failures, and successful initialization after failure.
The shared-MTP CTest generates a small Gemma4 target and Gemma4Assistant head.
It disables flash attention to expose the empty shared-KV softmax failure and
compares 16 greedy generated tokens with an independent baseline, including
real draft/verify/accept/rollback rounds. The independent-head fixture runs the
same generation comparison. An ordinary decoder can also be supplied alone
for the preservation checks.

For local recurrent `FULL`/`RS` classification and cleanup, supply a trained
recurrent target that supports snapshots and the generated independent head:

```sh
build/mtp-rollback/llamadart_speculative_api_test --rollback-types \
  /path/to/recurrent-target.gguf build/mtp-rollback/exit_teardown_test_mtp_model.gguf
```

This checks the actual supplied target with zero and two snapshots. The
unrelated head isolates probe behavior; it does not qualify that model family
for MTP generation.

This is component evidence, not MTP generation-quality qualification. A trained
MTP-family baseline-versus-speculative output comparison, packaged artifacts
and cross-platform qualification remain separate gates. Core consumer
preparation and old/new native version-skew checks remain required before ABI retirement.
