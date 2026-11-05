# v0.8.0-3 wrapper rebuild

This proposed wrapper rebuild completes the
[speculative API consolidation](speculative_api_consolidation.md) for
[#51](https://github.com/leehack/llamadart-native/issues/51). It keeps upstream
v0.8.0 (`8642541b255bfb5dc39cd351a178be09013a55d6`) and the carried patches
unchanged. This document describes the candidate contract, not a published or
qualified release.

## ABI migration

The constant-false `llama_dart_speculative_need_embd` export and header
declaration are removed. `llama_dart_speculative_need_embd_nextn` remains: MTP
sessions, including mixed strategies, return true; non-MTP and null sessions
return false. Initialization still configures target outputs through upstream.
The private handle remains opaque to callers. Generic strategy, rollback and
ownership behavior is unchanged.

The export gate requires 92 surviving exports and forbids 15 retired exports.
The fourteen MTP/n-gram exports retired in `v0.8.0-2` stay forbidden.
The owner manifest layout and hook contract version remain unchanged, with no
new ABI handshake. Normal header/archive digests change with the new bytes.
The coordinated core migration separately isolates llama-native downloads in
`native_bundles_v2`, as described below.

## Consumer delivery

Published core 0.17.5 and Apple companion 0.17.3 keep their original pinned
artifacts. Preserve all existing release files, tags and digests. Core 0.18 must
adopt the new owner release after publication with synchronized headers/bindings
and Apple checksums. Core preparation first enforces the existing checksum map
for the currently qualified XCFramework; the v0.8.0-3 pin follows owner publication.

The coordinated core change will use `native_bundles_v2` consistently for the
hook writer, Windows fallback discovery and Linux dependency copying, without
migrating the legacy cache. This separates the new query-free bundles from old
consumers' cache namespace. It changes neither the owner manifest nor the hook
contract version.

Overriding an old consumer to this artifact violates the existing requirement
for ABI- and symbol-compatible overrides. Upgrade the consumer first. The old
dynamic fallback eagerly checks the removed query and reports a typed
`LlamaUnsupportedException`. The late, potentially untyped lookup risk belongs
to the old Windows lazy code-asset factory when deliberately paired with an
incompatible override; it is not caused by fallback scanning. Actual
Windows/AOT evidence remains pending.

## Qualification and provenance

Source-only and synthesized-symbol tests cannot qualify native artifacts.
Before release, record Debug wrapper-contract CTests and exact-head owner CI,
then inspect every candidate wrapper/slice and header bundle for the export
contract. Verify the complete manifest, digests, source/upstream identity,
dependencies, patch markers and platform metadata. Keep the existing full
platform/backend matrix and Apple OS-floor gates.

Bind ordinary old delivery, new-consumer compatibility, incompatible-override
controls and trained speculative/state/lifetime checks to the actual candidate
library hashes. Actual Windows/AOT evidence is required for Windows claims.
Historical model/backend evidence is reusable only after source/byte comparison
and explicit scoped acceptance; a rebuilt wrapper needs fresh provenance.

Retain and hash the raw release-publication ZIP. Qualify those exact bytes and
use `publish_prepared.yml` for validation and separately authorized publication
without rebuilding. Verify public assets and the native tag against the retained
receipt. Downstream adoption follows owner publication; source merge alone
satisfies none of these artifact or consumer gates.

Native compilation, packaged export inspection, model/device checks and
publication remain pending for this source preparation.
