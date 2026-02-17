# llamadart-native

Native build and release pipeline for `llamadart` binaries.

## Purpose

This repository is responsible for:

- Building native `llamadart` binaries across platforms.
- Publishing release artifacts consumed by `llamadart` build hooks.
- Producing release metadata (`assets.json` + `SHA256SUMS`).
- Syncing release tag updates back to `llamadart` via pull request.

The Dart API/runtime stays in the main `llamadart` repository.

## Workflows

- `Native Build & Release` (`.github/workflows/native_release.yml`)
  - Manual dispatch.
  - Backend configurable per platform.
  - Produces release assets with backend suffixes.
  - Generates `assets.json` and `SHA256SUMS`.
- `Sync llamadart Hook` (`.github/workflows/sync_llamadart_hook.yml`)
  - Triggered by published release (or manual dispatch).
  - Opens PR in `leehack/llamadart` to update `hook/build.dart`.

## Platform Defaults

Default backend policy (can be overridden in workflow inputs):

- Android: `opencl`
- iOS/macOS: `metal`
- Linux: `vulkan`
- Windows: `vulkan`

## Release Asset Naming

Artifacts include backend suffix to support configurable profiles, e.g.:

- `libllamadart-linux-x64-vulkan.so`
- `libllamadart-linux-x64-cuda.so`
- `libllamadart-windows-x64-vulkan.dll`
- `libllamadart-android-arm64-opencl.so`

## Required Secrets

For hook-sync automation:

- `LLAMADART_REPO_TOKEN`: token with permission to push branch and open PR on `leehack/llamadart`.

## Repository Layout

- `.github/workflows/native_release.yml`: build + package + release.
- `.github/workflows/sync_llamadart_hook.yml`: post-release PR sync.
- `third_party/`: native CMake wrapper and platform build scripts.
- `scripts/generate_assets_manifest.sh`: builds `assets.json` + checksums.
- `scripts/update_llamadart_hook.sh`: updates tag and base URL in `hook/build.dart`.
- `docs/platform_backend_strategy.md`: default/configurable backend matrix.
