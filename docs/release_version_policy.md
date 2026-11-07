# Native Release Version Policy

`llamadart-native` has a stable distribution channel and an explicit
development channel. A release tag is immutable and identifies the native
wrapper source and archive set; it is not a substitute for upstream source
provenance.

## Accepted tags

| Purpose | Upstream `llama.cpp` ref | Native release tag | New release classification |
| --- | --- | --- | --- |
| Stable distribution | `vMAJOR.MINOR.PATCH` | the same `vMAJOR.MINOR.PATCH` | release |
| Stable wrapper-only rebuild | `vMAJOR.MINOR.PATCH` | `vMAJOR.MINOR.PATCH-N` | prerelease |
| Explicit nightly/development build | `bNNNN` | the same `bNNNN` | prerelease |
| Nightly wrapper-only rebuild | `bNNNN` | `bNNNN-N` | prerelease |
| Legacy nightly wrapper (read only) | `bNNNN` | `bNNNN-llamadart.N` | historical metadata varies |

`N` starts at 1 and increases numerically. For example, wrapper-only rebuilds
of upstream `v0.2.0` are `v0.2.0-1`, `v0.2.0-2`, and so on. The exact
upstream version therefore remains visible as the native tag prefix. Nightly
rebuilds follow the same compact rule: `b10545-1`, `b10545-2`, and so on.

The repository enforces this native release sequence numerically:

```text
v0.2.0, v0.2.0-1, v0.2.0-2, ...; then v0.2.1
```

`v0.2.0-1` is a valid SemVer prerelease and generic SemVer comparison places it
before `v0.2.0`. Consumers must therefore treat `-N` as this repository's
native rebuild counter, parse the native tag grammar for channel selection, and
read `llama_cpp_tag`/`llama_cpp_commit` for upstream provenance. GitHub's
`prerelease` field is additional presentation metadata, not the historical
channel authority.
The policy does not use build metadata because SemVer ignores it for
precedence, and it does not increment the patch because that would obscure the
exact upstream version prefix.

The policy check refuses an existing tag, a stable-channel rollback, or a
decreasing rebuild sequence for the same upstream line. A wrapper rebuild also
requires an existing upstream-aligned tag or lower rebuild on the same line and
fails if a newer stable upstream line has already been published. Select that
newer upstream stable release instead of inventing a different suffix.

## Channel behavior

- `latest` resolves `ggml-org/llama.cpp`'s latest GitHub Release and must resolve
  to an exact `vMAJOR.MINOR.PATCH` tag. Automatic discovery fails closed on a
  nightly, prerelease, or unrecognized tag.
- `bNNNN` remains available only as an explicit workflow input. Newly emitted
  nightly and wrapper-only releases are marked as GitHub prereleases, so they
  cannot replace the default stable release.
- Only an upstream-aligned stable release may update this repository's pinned
  `third_party/llama.cpp` submodule. Nightly and wrapper-only releases do not
  move the stable pin.
- Existing `bNNNN` and `bNNNN-llamadart.N` releases and archives remain
  immutable and supported for explicit consumption. New nightly rebuilds emit
  only `bNNNN-N`; the `-llamadart.N` form is never emitted again.
- Historical GitHub metadata is not retroactively rewritten. In particular,
  immutable releases `b10545` and `b10356-llamadart.1` both have
  `prerelease=false`. Consumers must still classify both as nightly from their
  tag grammar. The repository's live regression check preserves this contract.

## Orchestration and approval

- The stable detector runs only on manual dispatch, after `main` qualifies the
  upstream release, and detects unbuilt stable upstream candidates. It may
  dispatch the native release workflow, and only for an exact upstream-aligned
  stable `vMAJOR.MINOR.PATCH` release that has no corresponding native release.
  Every other discovery result stops without a dispatch.
- The automatic dispatch always supplies the exact upstream ref, its full 40-hex
  upstream commit, the identical native release tag, `smoke_policy=required`,
  and `publish_release=true`. It cannot select a nightly ref, a wrapper rebuild
  tag, or a skipped smoke.
- The automatic caller correlation identifier is
  `auto-stable/<native-tag>/<sha256-prefix>`, derived only from the upstream
  ref, upstream commit, and native tag. It is therefore identical on a retried
  dispatch and on every later detector run for the same release, so all
  attempts at one release correlate to a single central operation. The
  publication transaction itself stays bound to its own workflow run, so a new
  dispatch still cannot take over a previous run's partial state.
- Automation queries for unsettled `native_release.yml` runs while planning and
  again immediately before dispatch. The final check also reconfirms the exact
  upstream commit and release absence; any changed or ambiguous state fails or
  suppresses the dispatch. The workflow's own concurrency group serializes
  detection, while the native workflow's concurrency group and immutable
  publication checks remain the fail-closed backstop for a final API
  dispatch/listing race.
- Each detection uploads `native-discovery-report-<run-id>` containing a
  machine-readable `candidate`, `noop`, or `incompatible` status, the
  `dispatch`/`skip`/`fail` decision, the exact upstream ref/commit, the detector
  workflow head, the in-flight native release run count, and run metadata.
  Detection holds `contents: read` plus `actions: write` for the dispatch alone;
  it cannot push, publish, or move the submodule. The report and plan upload
  completes before any authorized dispatch is attempted.
- Nightly refs and wrapper-only rebuilds remain manual dispatches. Reuse or
  republication of an existing tag is rejected. Central `llamadart`
  orchestration coordinates native, Dart, and Web preparation for valid manual
  change sets.
- The explicit dispatch contract requires an exact `llama_cpp_tag`, its full
  40-hex `llama_cpp_commit`, an exact `native_release_tag`, `smoke_policy`, and
  stable caller `correlation_id`. Publication requires
  `smoke_policy=required`; explicit `skip` is allowed only for a non-publishing
  preparation run. Moving `latest` and `submodule` inputs are rejected.
- Native publication does not require dependency-pin PRs in companion
  repositories. Any companion code, asset, or pin change is independent and
  retains its own validation and approval boundary.

## Provenance contract

New `assets.json` manifests carry four distinct identities:

- `native_release_tag`: native artifact tag and archive suffix;
- `llama_cpp_tag`: requested upstream release ref;
- `llama_cpp_commit`: exact upstream source commit;
- `native_commit`: exact native wrapper/provenance commit targeted by the tag.

The historical `tag` field remains as a compatibility alias for
`native_release_tag`. Older immutable manifests that contain only `tag` remain
valid. New manifests and release notes also bind the caller correlation ID,
smoke policy/conclusion, exact workflow run, and workflow head SHA.

Every release is built with the llama.cpp patches of its native commit that
its llama.cpp still needs (`patches/llama.cpp/README.md`): a patch whose
`upstream_fix` text is in the upstream source is retired for that build, and
the Android artifact validation fails a release whose bundle lacks a needed
one. `llama_cpp_patches` in `assets.json` lists the `file`, `sha256` and
`tracking` issue of each needed patch, so it can be shorter than
`patches/llama.cpp/`, and is an empty list when none is needed.
`llama_cpp_commit` stays the upstream commit the patches apply to: with a
non-empty `llama_cpp_patches` the artifacts are that commit plus exactly those
patches, and a comparison against upstream `llama.cpp` at the same commit must
account for them. A rebuild that only adds, refreshes or drops a patch is a
wrapper-only rebuild for the tag grammar: the upstream ref is unchanged and
the rebuild counter increases.

Do not mutate, republish, or reuse a tag. For a wrapper-only fix, keep the same
`llama_cpp_tag`, choose the next policy-compliant native rebuild tag, and run the
full build matrix. Publication, downstream pin updates, and downstream releases
remain separate actions; only the exact stable upstream dispatch is automatic.

## Publication and recovery

Build, package, and candidate-validation jobs run with `contents: read` and do
not persist checkout credentials. Only the final publication job receives
`contents: write`; the later stable-submodule update is a separate narrowly
scoped write job and runs only after a successful upstream-aligned stable
publication.

Publication is an immutable, draft-first transaction:

1. Resolution commits the selected llama.cpp gitlink and uploads its Git bundle.
   Every build and packaging job imports that same owner commit, so the binary
   build identity, manifest and release tag name the same source. The complete
   build matrix then produces one same-run artifact containing release assets,
   their digest, and that exact provenance commit.
2. The final job verifies that commit and its `llama.cpp` tree entry, creates an
   annotated tag whose immutable message binds the transaction ID, pushes it
   without force, creates a draft release, and uploads only missing assets.
3. The draft becomes published only after the tag, release correlation fields,
   prerelease setting, and every asset digest and size match exactly.

If publication stops partway through, retry the failed job in the same workflow
run. An exact matching tag, draft, or already-published release is resumed or
accepted idempotently. Any different tag target, release body, workflow/artifact
correlation, classification, asset set, digest, or size fails closed. The
workflow never force-moves a tag, replaces an asset, edits a mismatched release,
or repairs a partial published release by mutation. A new workflow dispatch is
a different transaction and cannot take over partial state from the old run.
This also applies when failure occurs immediately after the tag push: an orphan
tag resumes only when its annotated transaction marker matches the same run.
Unmarked tags and different transaction markers are ambiguous and fail closed.

### Publishing a qualified preparation without rebuilding

A preparation dispatched with `publish_release=false` cannot change its original
inputs on rerun. Once its exact bytes are qualified, dispatch
`publish_prepared.yml` from owner `main` with these original values:

- `run_id`, `artifact_id` and the raw publication artifact ZIP `artifact_sha256`;
- `workflow_head_sha`, `native_commit`, `upstream_ref`, `upstream_commit`;
- `native_release_tag`, `correlation_id`, and explicit `publish_approved`.

The default `publish_approved=false` validates without a release/tag mutation.
Set it to `true` only after publication is authorized and the exact artifact
qualification is complete. Repository/ref restrictions, a successful same-owner
main `native_release.yml` dispatch, every current build/OS-floor/package/smoke
job, producer timestamps and artifact API digests are checked before publication.
Failed, incomplete, foreign, fork, unexpected-skip or incomplete-matrix runs are
refused. The immutable publication ZIP must match the supplied digest; only its
expected regular files are extracted. The imported source bundle must name the
exact native commit and upstream gitlink. Every asset and checksum is rechecked,
and the reconstructed preparation contract must equal the GitHub-produced
same-run `native-release-result` artifact. Unknown source-result schemas or
changed gate layouts fail closed until the maintained contract is reviewed.

Both native release entrypoints hold the same workflow-level
`native-release-main` concurrency group with `cancel-in-progress: false`.
Serialization spans history admission, building or prepared-byte verification,
and final publication across tags and refs. The literal group also matches
historical main-run retries. Nonpublishing preparations share this conservative
serialization; the owner keeps one active native release transaction at a time.

Immediately before publication, the job reads every current remote tag and
release page and applies the maintained `validate_history` ordering and wrapper
lineage policy. A newer stable release/rebuild, missing earlier wrapper lineage,
or failed/incomplete history read blocks a new publication. An exact marked
tag-only or partial-draft transaction may resume only while current history
still permits it. A fully published exact transaction can be verified as a
read-only no-op after newer releases; its report records that new publication
would now be blocked. Unmarked/mismatched tags, release metadata and assets can
never use that exception, and it never invokes the publisher.

The dedicated publication job reuses `release_publication.py`; it does not build
or alter an asset. Prerelease classification comes from the version policy,
including `true` for wrapper rebuilds. The original build run/head, artifact
digest and correlation still determine the transaction ID. Its report records
the new publication executor separately, without relabeling the source build.
The same draft-first collision, exact-state retry and immutable-asset rules
above apply. Retry with the identical original values if publication stops;
a different source transaction cannot take over partial state.

Before packaging, `smoke_policy=required` downloads the same-run Linux x64 core
artifact, loads `libllamadart.so` with only packaged sibling libraries visible,
and calls the model-free wrapper API-version probe. A failed required smoke
blocks packaging and publication. The final
`native-release-result-<run-id>` JSON artifact provides the exact workflow and
release metadata, publication artifact digest, manifest/checksum digests and
entries, complete required/actual bundle coverage, and smoke conclusion for
central verification. The result repeats both `native_release_tag` and legacy
`tag` and keeps the upstream ref/commit and native commit separate.

## Downstream coordination

Consumers must classify the native tag independently from both the upstream ref
and GitHub's historical `prerelease` field:

- `llamadart` issue #393 should accept stable `vMAJOR.MINOR.PATCH`, stable
  wrapper rebuilds in `vMAJOR.MINOR.PATCH-N` form, historical `bNNNN`, and
  compact nightly rebuilds in `bNNNN-N` form. Historical
  `bNNNN-llamadart.N` remains read-only compatibility. It
  should use `llama_cpp_tag`/`llama_cpp_commit` for upstream provenance rather
  than deriving them from the native archive suffix.
- `llama-web-bridge` issue #61 should use `vMAJOR.MINOR.PATCH` as its automatic
  stable update channel, retain `bNNNN` only for explicit development input,
  and refuse an accidental stable-to-nightly transition. Native
  `vMAJOR.MINOR.PATCH-N` and `bNNNN-N` rebuild tags retain the exact upstream
  prefix and do not change the bridge's upstream `llama.cpp` ref. Neither
  companion repository needs a dependency-pin PR as a prerequisite for native
  publication.

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [ggml-org distribution and release-channel policy](https://github.com/ggml-org/ggml/discussions/1579)
- [`llamadart` semantic native-tag synchronization issue #393](https://github.com/leehack/llamadart/issues/393)
- [`llama-web-bridge` semantic update/publish issue #61](https://github.com/leehack/llama-web-bridge/issues/61)
