# AGENTS.md

Guidance for coding agents working in `llamadart-native`.

## Scope and Ownership

- This repository owns native build, packaging, and release artifacts consumed by `llamadart`.
- Upstream source is vendored as submodules under `third_party/`.
- `llamadart` repository consumes published release bundles from this repo.

## Related Repositories

Common maintainer sibling layout:

```text
../llamadart
../llamadart-native
../llama-web-bridge
../llama-web-bridge-assets
```

This layout is a convention only. Verify paths before use.

## Build Commands

```bash
git submodule update --init --recursive
python3 tools/build.py list
python3 tools/build.py apple --target macos-arm64
python3 tools/build.py linux --arch x64
python3 tools/build.py android --abi all
python3 tools/build.py windows --arch x64
```

Optional Linux container build:

```bash
./tools/docker_build_linux.sh --arch x64 --jobs 8
```

## Release Workflows

- `.github/workflows/native_release.yml`
  - Manual native build + release publish.
- `.github/workflows/auto_native_release.yml`
  - Scheduled/manual dispatcher when upstream `llama.cpp` tag advances.

## Change Boundaries

- Prefer updating submodule refs over patching vendored upstream code directly.
- Keep wrapper/runtime integration changes under `src/` and top-level build config.
- Keep release manifest logic in `scripts/`/`tools/`.

## Cross-Repo Handoff to `llamadart`

After publishing here:

1. Update `llamadart` native tag pin.
2. Sync header bundle + regenerate bindings:
   `tool/native/sync_native_headers_and_bindings.sh --tag <tag>`
3. Run `dart analyze` and tests in `llamadart` before merge.
