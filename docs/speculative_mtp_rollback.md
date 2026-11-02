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

The upstream `common_context_can_seq_rm` probe clears memory, decodes two dummy
tokens, attempts partial removal, then clears memory again. A fresh-memory
precondition makes this safe for a caller-owned target. Checking only sequence
zero would discard other populated sequences. `llama_memory_can_shift` does
not establish rollback support, and constructing an unrelated scratch context
does not establish the supplied target's configuration.

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
An ordinary decoder can also be supplied alone for the preservation checks.

This is component evidence, not MTP generation-quality qualification. A trained
MTP-family model, greedy baseline-versus-speculative output comparison,
recurrent `FULL` and bounded-snapshot `RS` coverage, packaged artifacts and
cross-platform qualification remain separate gates. Core consumer preparation
and old/new native version-skew checks remain required before ABI retirement.
