# Windows consumer ABI qualification

This owner-side CPU lane qualifies the existing `v0.8.0-3` preparation through
real Windows consumers before publication. It does not rebuild native code,
publish a release, move a core pin, or qualify GPU/trained MTP behavior.

`inputs.json` binds this campaign to merged core
`7ef0eadc4826ab5a0bbe3e90cea177c48e9acefa` (the audited `33ad515` tree), the
actual pub.dev `llamadart 0.17.5` source archive, old native `v0.8.0-1`,
Flutter 3.47.1 / Dart 3.13.1, one public revision-pinned Qwen2.5 model, and the
original owner workflow head/upstream/tag/correlation. The source archive was
independently refreshed from official pub.dev: 4,191,094 bytes and 1,695 regular
files, matching the previously inspected published source. The generated native
provenance commit is a separate required input; never infer it from the workflow
head. Candidate artifact ID and raw ZIP hash are supplied from the independently
accepted terminal preparation receipt, not from a mutable release/latest URL.

## Cases

| Case | Required observation |
| --- | --- |
| `old-default` | Unmodified published source and ordinary exact old hook acquisition; public greedy baseline and active ngram-simple, full output parity, positive drafted/accepted counts, unload/reload/reuse and disposal. Old code assets remain selected with mixed legacy/v2 cache decoys. |
| `new-candidate` | Exact merged core with the documented local candidate-directory selector; same public behavior, absent `need_embd`, retained `nextn(null)=false`, and v2 discovery. |
| `old-incompatible-candidate` | Deliberately unsupported old consumer/candidate selector; actual missing-query error after public generation enters the old speculative path. No typed-error requirement is invented. |
| `old-aot`, `new-aot` | Whole `dart build cli` bundles copied outside every source/app root, distinct empty CWD, system-only PATH, no runtime override environment, identical before/after inventories and exact loaded module/symbol-owner hashes inside the copied bundle. |

All applications select only llama.cpp and CPU modules. Ordinary old selection
has no native-path/tag override. Only candidate applications add the local-path
selector. These application pubspecs and resolved locks/configs are retained;
neither old nor new core source is patched. Public generation uses a 128-token
budget, greedy sampling and two CPU threads. The output texts, hashes, actual
speculative counters and teardown outcomes are retained. A zero-draft result is
a failure. On the exact old source, successful active speculative generation
necessarily reaches `speculativeApi.needEmbd(speculativeSession)`; the pinned
source proof and actual provider map bind that inference. The separate raw FFI
probe only verifies export presence/absence, symbol ownership and null nextn;
it does not stand in for public generation or claim live MTP behavior.

The model-free public cache resolver also records the old missing-assets/v2-only
case honestly. A best-effort directory result is not claimed as a typed native
loader refusal. Actual loaded providers are checked separately.

## Execution and evidence

The workflow has a model-free PR source-check job: PowerShell AST parsing,
synthetic Python tests, and a no-network PowerShell staging regression using
the production download function and safe extractor. Download archives stay
under `stage/downloads`, separate from their fresh extraction directories.
Its real qualification job is restricted to a manual
owner/main dispatch and an exact expected executor SHA. It has only
`contents:read` and `actions:read`; its normal Actions token reads this owner's
own exact publication artifact. No publication/provider credentials are used.

After review and merge, root supplies the accepted artifact receipt:

```sh
gh workflow run windows_consumer_qualification.yml --repo leehack/llamadart-native --ref main \
  -f expected_executor_sha=REVIEWED_OWNER_EXECUTOR_SHA \
  -f core_sha=7ef0eadc4826ab5a0bbe3e90cea177c48e9acefa \
  -f owner_run_id=42040929247 \
  -f owner_artifact_id=ACCEPTED_PUBLICATION_ARTIFACT_ID \
  -f owner_artifact_sha256=ACCEPTED_RAW_ZIP_SHA256 \
  -f owner_workflow_head_sha=703c0f51fe9f8ebfa941c70504ad0b13c7ca79cb \
  -f owner_native_commit=ACTUAL_GENERATED_NATIVE_COMMIT
```

Intake calls only `publish_prepared_release.prepare`, never its publishing CLI
or `publish` function. That maintained validator binds completed owner/main
workflow/job gates, original attempt, producer timestamps, exact artifact digest,
source bundle/gitlink, required smoke, manifest/checksums and same-run result.
The fixed trusted campaign correlation is checked rather than inferred from the
result being verified. Raw archives remain in fresh scratch; no retry, artifact
deletion or fallback follows failure. Additional extraction rejects unsafe paths,
case collisions, links/reparse points and oversized inventories before writes.

The source-bound core observer and AOT helpers are copied/read from that exact
checkout; their historical registry inputs and 16-case matrix remain untouched.
The whole AOT bundle uses `load_stock.exe` because that is the reviewed helper's
existing target layout. The consumer payload itself is `payload/consumer.dart`.

Every child has a timeout, raw logs, fresh temporary/Pub state and process-tree
cleanup on timeout. The first failure is sealed too. Upload names include core,
owner run, qualification run and attempt; overwrite is disabled. Upload success
is not qualification. Independently verify the downloaded sealed directory with
an expected identity receipt supplied separately from the downloaded evidence:

```sh
python tools/qualification/windows_consumer/verify_evidence.py verify EVIDENCE_DIR \
  --expected ROOT_TRUSTED_EXPECTED_IDENTITY.json
```

The expected receipt has `executorSha`, `coreSha`, `ownerRunId`, `ownerArtifactId`,
`ownerArtifactSha256`, `ownerWorkflowHeadSha`, `ownerNativeCommit`, `runId` and
`runAttempt`. Full source SHAs, raw artifact SHA256 and canonical positive ID
strings are required. Keep the publication ZIP and original preparation receipt
for independent artifact verification; the small consumer evidence does not
replace those bytes. The verifier checks the complete seal, five-case inventory,
source/runtime identity, raw logs, provider ownership, active speculation,
cleanup and complete AOT distribution.

Run local synthetic checks without SDK/native/model work:

```sh
python -B -m unittest discover -s tests -p test_windows_consumer_qualification.py
```

This prepublication selector is diagnostic. After publication and final
core/header/binding/SwiftPM adoption, repeat the ordinary new default-hook AOT
control without the local-path overlay. Existing owner live-nextn CTests and
trained speculative/model gates remain separately scoped evidence. Neither this
source PR nor synthetic verifier tests claim actual Windows execution.
