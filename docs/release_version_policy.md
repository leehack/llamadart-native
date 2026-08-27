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

- Scheduled automation detects unbuilt stable upstream candidates daily. It may
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
  dispatch and on every later scheduled run for the same release, so all
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

1. The complete build matrix produces one same-run artifact containing release
   assets, their digest, and the exact provenance commit.
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
