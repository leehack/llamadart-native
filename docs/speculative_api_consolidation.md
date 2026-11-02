# Generic speculative API consolidation

This preparation for [#51](https://github.com/leehack/llamadart-native/issues/51)
removes fourteen unused MTP/n-gram C exports. The replacement is the existing
`llama_dart_speculative_*` family. Both legacy queries
`llama_dart_speculative_need_embd` and `llama_dart_speculative_need_embd_nextn`
remain exported with their existing answers for older Dart consumers
([#48](https://github.com/leehack/llamadart-native/issues/48)). No generic
version handshake or new runtime pin is introduced.

## Removed exports

| Prefix | Removed suffixes |
| --- | --- |
| `llama_dart_mtp_` | `init`, `init_with_draft_model`, `free`, `get_draft_context`, `begin`, `process_batch`, `draft`, `accept` |
| `llama_dart_ngram_` | `simple_init`, `free`, `begin`, `process_batch`, `draft`, `accept` |

`llama_dart_sampler_sample_and_accept_n` remains. The export validator requires
both compatibility queries and rejects a partial build that still exports any
of the fourteen retired functions. Direct clients of those functions must
migrate before adopting the new artifact. Generated Dart bindings and pinned
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

The validated consumer migration must land before this owner retirement is
pushed or merged. Fresh candidate artifact tests must cover old/new Dart binding
paths, Gemma MTP and n-gram baseline/output controls. Existing export-filtered
fixtures are negative/compatibility controls, not the retired owner artifact.

Immutable publication, downstream repin/binding regeneration, representative
DSpark/DFlash/state/output checks and wider platform qualification remain
separate release gates. This source preparation does not close #51 or authorize
publication.
