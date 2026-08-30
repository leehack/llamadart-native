# Contributing to llamadart-native

Thanks for contributing.

## Purpose

`llamadart-native` builds and publishes native runtime bundles consumed by
`llamadart` hooks.

## Prerequisites

- Python 3
- CMake + Ninja
- Platform toolchains (Android NDK, Xcode, MSVC, etc.)
- Git submodules initialized

## Setup

```bash
git clone https://github.com/leehack/llamadart-native.git
cd llamadart-native
git submodule update --init --recursive
python3 tools/build.py list
```

## Local Build Examples

```bash
python3 tools/build.py apple --target macos-arm64
python3 tools/build.py linux --arch x64
python3 tools/build.py android --abi arm64-v8a --backend vulkan
python3 tools/build.py windows --arch x64 --backend vulkan
```

## Release Process

1. Ensure working tree is clean and submodules are in intended state.
2. Select a policy-compliant version from
   [`docs/release_version_policy.md`](docs/release_version_policy.md): use an
   exact `vMAJOR.MINOR.PATCH` for stable distribution, or an explicit `bNNNN`
   only for a nightly/development build. For a wrapper-only stable
   rebuild, preserve the upstream version prefix and append the next native
   rebuild number, such as `v0.2.0-1` for upstream `v0.2.0`.
   Nightly rebuilds use the same compact suffix, such as `b10545-1` for
   upstream `b10545`; `bNNNN-llamadart.N` remains read-only legacy syntax.
3. For an exact stable upstream release, let `Detect and Prepare Stable Native
   Candidate` prepare it. Scheduled automation dispatches `Native Build &
   Release` at most once per detector run, with the exact
   upstream ref/commit, the identical native tag, `smoke_policy=required`,
   `publish_release=false`, and a retry-stable `auto-stable/<tag>/<digest>`
   correlation identifier.
4. Review the machine-readable report and successful preparation result. The
   repository owner then approves the exact candidate at **Actions > Native
   Build & Release > Run workflow** from the default branch, copies the exact
   upstream commit, `native_source_sha`, native tag, smoke policy, and
   correlation, and explicitly sets `publish_release=true`. Nightly and wrapper
   rebuilds use this same manual approval path.
5. Verify release assets (`assets.json`, `SHA256SUMS`, per-target bundles) and
   confirm that the native tag, upstream ref/commit, and native commit are
   distinct and correct in the manifest and release notes. Download the final
   release-result JSON and verify its workflow/release metadata, digests,
   checksum entries, bundle coverage, correlation, and smoke conclusion.

Never reuse or republish a release tag. Automatic preparation is stable-only and
is suppressed when its planning or final pre-dispatch query observes a native
release run queued or in progress;
newly emitted nightly and wrapper rebuild releases are GitHub prereleases and
require explicit selection. Historical metadata is immutable: `b10545` and
`b10356-llamadart.1` have `prerelease=false`, so consumers must parse tag grammar
instead of using GitHub classification alone. Candidate builds and packaging
are read-only; write access is limited to final publication and the separate
post-publication stable-submodule update. Issue #60 separately tracks binding
the final publication job to a protected environment; this process does not
assume that repository setting exists. If publication partially fails, the
repository owner may rerun the failed job in the same approved workflow run so
exact matching state can resume;
mismatches fail closed and tags/assets are never replaced. The annotated tag's
transaction marker protects even tag-only partial state from takeover by a
different workflow run. Companion
repositories do not need dependency-pin PRs as a native publication
prerequisite. Dispatching, publishing, and any downstream code, asset, or pin
changes are separate maintainer actions.

## Repository Boundaries

- Native build/release logic belongs here.
- Dart API/runtime behavior belongs in `llamadart`.
- Web bridge runtime belongs in `llama-web-bridge` +
  `llama-web-bridge-assets`.
