# Native Release Version Policy

`llamadart-native` has a stable distribution channel and an explicit
development channel. A release tag is immutable and identifies the native
wrapper source and archive set; it is not a substitute for upstream source
provenance.

## Accepted tags

| Purpose | Upstream `llama.cpp` ref | Native release tag | GitHub classification |
| --- | --- | --- | --- |
| Stable distribution | `vMAJOR.MINOR.PATCH` | the same `vMAJOR.MINOR.PATCH` | release |
| Stable wrapper-only rebuild | `vMAJOR.MINOR.PATCH` | `vMAJOR.MINOR.PATCH-N` | prerelease |
| Explicit nightly/development build | `bNNNN` | the same `bNNNN` | prerelease |
| Nightly wrapper-only rebuild | `bNNNN` | `bNNNN-llamadart.N` | prerelease |

`N` starts at 1 and increases numerically. For example, wrapper-only rebuilds
of upstream `v0.2.0` are `v0.2.0-1`, `v0.2.0-2`, and so on. The exact
upstream version therefore remains visible as the native tag prefix.

The repository enforces this native release sequence numerically:

```text
v0.2.0, v0.2.0-1, v0.2.0-2, ...; then v0.2.1
```

`v0.2.0-1` is a valid SemVer prerelease and generic SemVer comparison places it
before `v0.2.0`. Consumers must therefore treat `-N` as this repository's
native rebuild counter, use the GitHub prerelease classification for channel
selection, and read `llama_cpp_tag`/`llama_cpp_commit` for upstream provenance.
The policy does not use build metadata because SemVer ignores it for
precedence, and it does not increment the patch because that would obscure the
exact upstream version prefix.

The policy check refuses an existing tag, a stable-channel rollback, or a
decreasing rebuild sequence for the same upstream line. A wrapper rebuild also
fails if a newer stable upstream line has already been published. Select that
newer upstream stable release instead of inventing a different suffix.

## Channel behavior

- `latest` resolves `ggml-org/llama.cpp`'s latest GitHub Release and must resolve
  to an exact `vMAJOR.MINOR.PATCH` tag. Automatic discovery fails closed on a
  nightly, prerelease, or unrecognized tag.
- `bNNNN` remains available only as an explicit workflow input. Nightly and
  wrapper-only releases are marked as GitHub prereleases, so they cannot replace
  the default stable release.
- Only an upstream-aligned stable release may update this repository's pinned
  `third_party/llama.cpp` submodule. Nightly and wrapper-only releases do not
  move the stable pin.
- Existing `bNNNN` and `bNNNN-llamadart.N` releases and archives remain
  immutable and supported for explicit consumption.

## Provenance contract

New `assets.json` manifests carry four distinct identities:

- `native_release_tag`: native artifact tag and archive suffix;
- `llama_cpp_tag`: requested upstream release ref;
- `llama_cpp_commit`: exact upstream source commit;
- `native_commit`: exact native wrapper/provenance commit targeted by the tag.

The historical `tag` field remains as a compatibility alias for
`native_release_tag`. Older immutable manifests that contain only `tag` remain
valid. The release notes repeat all four identities.

Do not mutate, republish, or reuse a tag. For a wrapper-only fix, keep the same
`llama_cpp_tag`, choose the next policy-compliant native rebuild tag, and run the
full build matrix. Workflow dispatch, publication, downstream pin updates, and
downstream releases remain separate maintainer actions.

## Downstream coordination

Consumers must classify the native tag independently from the upstream ref:

- `llamadart` issue #393 should accept stable `vMAJOR.MINOR.PATCH`, stable
  wrapper rebuilds in `vMAJOR.MINOR.PATCH-N` form, historical `bNNNN`, and
  historical nightly rebuilds. It
  should use `llama_cpp_tag`/`llama_cpp_commit` for upstream provenance rather
  than deriving them from the native archive suffix.
- `llama-web-bridge` issue #61 should use `vMAJOR.MINOR.PATCH` as its automatic
  stable update channel, retain `bNNNN` only for explicit development input,
  and refuse an accidental stable-to-nightly transition. Native
  `vMAJOR.MINOR.PATCH-N` rebuild tags retain the bridge's exact upstream
  `vMAJOR.MINOR.PATCH` prefix and do not change its upstream `llama.cpp` ref.

## References

- [Semantic Versioning 2.0.0](https://semver.org/)
- [ggml-org distribution and release-channel policy](https://github.com/ggml-org/ggml/discussions/1579)
- [`llamadart` semantic native-tag synchronization issue #393](https://github.com/leehack/llamadart/issues/393)
- [`llama-web-bridge` semantic update/publish issue #61](https://github.com/leehack/llama-web-bridge/issues/61)
