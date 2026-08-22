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
   [`docs/release_version_policy.md`](docs/release_version_policy.md): use
   `latest`/`vMAJOR.MINOR.PATCH` for stable distribution, or an explicit
   `bNNNN` only for a nightly/development build. For a wrapper-only stable
   rebuild, use the next-patch prerelease form such as
   `v0.2.1-llamadart.1` for upstream `v0.2.0`.
3. Run `Native Build & Release` workflow:
   `.github/workflows/native_release.yml`
4. Verify release assets (`assets.json`, `SHA256SUMS`, per-target bundles) and
   confirm that the native tag, upstream ref/commit, and native commit are
   distinct and correct in the manifest and release notes.
5. In `llamadart`, sync to the new native release tag and regenerate bindings.

Never reuse or republish a release tag. Automatic discovery is stable-only;
nightly and wrapper rebuilds are GitHub prereleases and require explicit
selection. Dispatching, publishing, and changing downstream pins are separate
maintainer actions.

## Repository Boundaries

- Native build/release logic belongs here.
- Dart API/runtime behavior belongs in `llamadart`.
- Web bridge runtime belongs in `llama-web-bridge` +
  `llama-web-bridge-assets`.
