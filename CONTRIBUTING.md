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
2. Run `Native Build & Release` workflow:
   `.github/workflows/native_release.yml`
3. Verify release assets (`assets.json`, `SHA256SUMS`, per-target bundles).
4. In `llamadart`, sync to the new native release tag and regenerate bindings.

## Repository Boundaries

- Native build/release logic belongs here.
- Dart API/runtime behavior belongs in `llamadart`.
- Web bridge runtime belongs in `llama-web-bridge` +
  `llama-web-bridge-assets`.
