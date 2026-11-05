# Generic speculative API consolidation

The consolidation for [#51](https://github.com/leehack/llamadart-native/issues/51)
retires fifteen C exports in total. `v0.8.0-2` removed fourteen unused MTP/n-gram
family exports in favor of the existing `llama_dart_speculative_*` family.
The proposed [v0.8.0-3 wrapper rebuild](v080_3_wrapper_rebuild.md) also removes
`llama_dart_speculative_need_embd`, whose answer was always false.
`llama_dart_speculative_need_embd_nextn` remains exported: it reports true for
MTP, including mixed MTP strategies, and false for non-MTP and null sessions
([#48](https://github.com/leehack/llamadart-native/issues/48)). Upstream
configures required target outputs during initialization. No generic version
handshake is introduced.

## Removed exports

| Prefix | Removed suffixes |
| --- | --- |
| `llama_dart_mtp_` | `init`, `init_with_draft_model`, `free`, `get_draft_context`, `begin`, `process_batch`, `draft`, `accept` |
| `llama_dart_ngram_` | `simple_init`, `free`, `begin`, `process_batch`, `draft`, `accept` |
| `llama_dart_speculative_` | `need_embd` (v0.8.0-3) |

`llama_dart_sampler_sample_and_accept_n` remains. The export validator requires
`llama_dart_speculative_need_embd_nextn` and rejects a partial build that still
exports any of the fifteen retired functions. Direct clients of those functions
must migrate before adopting the new artifact. Generated Dart bindings and pinned
artifact integration tests change only when the replacement artifact is adopted.

## Parameter mapping

Use `llama_dart_speculative_init(target_model, draft_model, target_context,
context_params, &params)`, then the generic free/get-draft-context/begin/
process-batch/draft/accept functions.

- MTP uses `params.type_names = "draft-mtp"`. A separate head is `draft_model`;
  pass null to use the target model's bundled MTP head. Map the previous draft
  maximum/minimum/probability and backend-sampling controls to their existing
  generic fields.
- N-gram uses `params.type_names = "ngram-simple"`. Map the previous
  `ngram_size` to `ngram_size_n`, and its initialization `draft_token_max` to
  `ngram_size_m`. The old constructor's nonpositive fallbacks were 12 and 48;
  set those explicitly to preserve those defaults. Generic defaults remain
  upstream defaults. The per-call draft maximum and output capacity still
  independently bound returned tokens.
- Preserve the generic API's documented negative sentinels for nullable numeric
  controls. A zero-initialized structure does not express every upstream default.

The [MTP rollback contract](speculative_mtp_rollback.md) remains unchanged:
initialize against empty target memory in every sequence before prompt ingest;
reject unsupported rollback and clean failed draft contexts. Draftless/n-gram
strategies keep their populated-target behavior and checkpoint fallback. No
upstream source or generic strategy implementation is changed by this removal.

## Lifetime and tests

MTP and n-gram now share the generic allocation/destructor family. Owner and
process cleanup retain stage ordering, generation claims, callback delegation,
concurrent/duplicate frees and address reuse for the surviving TTS and generic
families. Tests of the former separate payloads are ported to the generic
strategy/handle paths. Registry-only tests construct real draftless state in a
test helper; model-backed tests separately verify public initialization,
strategy inheritance and actual context cleanup.

Draft bounds, invalid sequence IDs, accept-before-draft, repeated acceptance,
failed-draft reset, malformed process batches and stale-error clearing remain
covered on the surviving API. The native rollback fixtures and trained-model
consumer controls qualify different things; model-free fixtures alone do not
establish output accuracy or speed.

## Delivery gates

Ordinary published core 0.17.5 and Apple companion 0.17.3 delivery remains on
its original tag/checksum-pinned artifacts. Keep `v0.8.0-2` and earlier release
files, tags and digests unchanged. Core 0.18 adopts the query-free artifact only
after owner publication, with synchronized headers and generated bindings.
Core preparation first enforces the existing checksum map for the currently
qualified XCFramework, then adopts the v0.8.0-3 pin after owner publication.

The coordinated core migration will isolate llama-native bundles in
`native_bundles_v2` across the hook writer, Windows fallback discovery and Linux
dependency copying, without migrating legacy caches. Ordinary old core and
SwiftPM delivery keeps its historical pins and cache namespace. The owner
manifest layout and hook contract version remain unchanged; no new handshake
is introduced.

An old consumer's explicit tag/path/manual-library override to a query-free
artifact is an unsupported, ABI-incompatible pairing: existing override guidance
requires ABI- and symbol-compatible libraries. Upgrade the consumer before using
that artifact. The old dynamic fallback eagerly checks the query and reports
a typed `LlamaUnsupportedException`. The old Windows lazy code-asset factory
can instead look up the symbol late when deliberately given an incompatible
override, so that path has no guaranteed typed refusal. This risk is not caused
by fallback scanning. Actual Windows/AOT verification remains pending.

Source checks require exactly 92 required and 15 retired exports, the selected
query absent from declarations/definitions/private fields, and the retained
nextn producer and null behavior unchanged. Debug wrapper-contract CTests must
exercise initialization, rollback, generation, ownership and cleanup. Every
candidate wrapper and Apple slice must omit all retired exports and retain all
required exports; header bundles, source identity, manifest inventory and file
digests must match the exact candidate.

Fresh candidate consumer tests must cover ordinary old pinned delivery, supported
new-consumer paths and explicit incompatible overrides separately, including
actual Windows/AOT paths where claimed. Trained Gemma MTP, Qwen3 DFlash/DSpark,
n-gram output controls, and state/cancellation/reuse/teardown checks must bind
the loaded library hashes to that candidate. Export-filtered fixtures are
negative/compatibility controls, not evidence for a packaged owner artifact.

Immutable publication, downstream repin/binding regeneration, independent
exact-head review and wider platform qualification remain separate release
gates. This source preparation does not close #51 or authorize publication.
