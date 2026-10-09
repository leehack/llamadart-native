# llamadart-native

Native build and release pipeline for `llamadart` binaries.

## Purpose

This repository is responsible for:

- Building native `llamadart` binaries across platforms.
- Publishing release artifacts consumed by `llamadart` build hooks.
- Producing release metadata (`assets.json` + `SHA256SUMS`).

The Dart API/runtime stays in the main `llamadart` repository.

## Workflow

- `Native Build & Release` (`.github/workflows/native_release.yml`)
  - Exact dispatch, manual or automatic for an upstream-aligned stable
    candidate, with an exact upstream ref/40-hex commit, exact native tag, smoke
    policy, and caller correlation identifier.
  - Builds one full backend set per platform/arch target.
  - Fails when any enabled backend in that target fails.
  - Publishes per-target native assets (Apple consolidated, others split core/backend libs).
  - Generates `assets.json` and `SHA256SUMS`.
  - Pins every build to one resolved `llama.cpp` commit and publishes a source
    tag whose submodule entry identifies that exact commit.
- `Auto Stable Native Release` (`.github/workflows/auto_native_release.yml`)
  - Daily schedule plus an optional manual detection run.
  - Resolves the latest stable upstream `ggml-org/llama.cpp`
    `vMAJOR.MINOR.PATCH` release tag and fails closed on any other shape.
  - Dispatches `Native Build & Release` only for an exact unbuilt stable
    upstream release, with the exact upstream ref/40-hex commit, the exact
    upstream-aligned native tag, `smoke_policy=required`, `publish_release=true`,
    and a deterministic `auto-stable/<tag>/<digest>` correlation identifier that
    is stable across retries.
  - Checks for queued or in-progress native release runs both while planning and
    immediately before dispatch, suppressing the dispatch when one is
    observable. It never publishes assets or mutates this repository itself.
  - Uploads `native-discovery-report-<run-id>` with a machine-readable
    `candidate`, `noop`, or `incompatible` status, the `dispatch`/`skip`/`fail`
    decision, and the exact upstream tag and commit before any dispatch.

## Native Version Management

Stable upstream `vMAJOR.MINOR.PATCH` releases are the default distribution
channel. Historical and new `bNNNN` nightlies remain available only through an
explicit workflow input. The published tag is the native version contract
consumed by downstream package hooks and Swift Package manifests.
`llama_cpp_tag` selects the exact upstream `llama.cpp` release ref to build and
`llama_cpp_commit` pins its expected full commit SHA; moving aliases such as
`latest` and `submodule` are not dispatch inputs.
`native_release_tag` selects the `llamadart-native` release tag and archive
suffix and is always explicit.

When changing the upstream `llama.cpp` version:

1. `Auto Stable Native Release` detects a new stable upstream tag daily and
   dispatches `Native Build & Release` for it automatically. Run it manually to
   pick the change up sooner.
2. For a policy-compliant nightly `bNNNN` ref or wrapper-only rebuild tag,
   dispatch `Native Build & Release` manually with an independently resolved
   exact `llama_cpp_tag`/`llama_cpp_commit`, an exact `native_release_tag`,
   `smoke_policy=required`, and a stable `correlation_id`. Reuse or republication
   of an existing tag is rejected.
3. Verify the release contains per-platform native archives,
   `llamadart-native-apple-xcframework-<tag>.zip`, `assets.json`, and
   `SHA256SUMS`.

Native publication does not require dependency-pin PRs in companion
repositories. Any downstream code, asset, or pin change remains an independent
change with its own validation and approval boundary.

When rebuilding the wrapper without changing the upstream `llama.cpp` ref,
dispatch `Native Build & Release` with the same `llama_cpp_tag` and a new
`native_release_tag`. A rebuild of stable upstream `v0.2.0` uses
`v0.2.0-1`; a rebuild of nightly `b9873` uses `b9873-1`. Historical
`b9873-llamadart.1` tags remain readable compatibility inputs but are never
emitted for new releases. Do not republish different source under an existing
tag; downstream caches are tag-keyed and the GitHub source tag should identify
the native wrapper commit that produced the assets. Publish mode rejects tag
collisions, stable rollbacks, and decreasing rebuild sequences.

Consumers determine stable versus nightly from the native tag grammar, not from
GitHub's `prerelease` field alone. Newly emitted nightlies and wrapper rebuilds
are prereleases, but immutable historical releases `b10545` and
`b10356-llamadart.1` have `prerelease=false` and remain nightly inputs.

`assets.json` records `native_release_tag`, the requested `llama_cpp_tag`, its
resolved `llama_cpp_commit`, and the `native_commit` targeted by the published
source tag. It also records the caller correlation, smoke policy/conclusion,
and exact workflow run/head. The historical `tag` field remains as a
compatibility alias for the native tag. Consumers can therefore verify the
built source independently of a moving ref or release label.

Candidate builds and packaging are read-only. The final publication job alone
can create the transaction-marked annotated tag and draft-first release; its
assets are published only after exact digest and provenance checks. Retry a
partial publication by rerunning the failed job in the same workflow run. Exact
matching partial state—including a tag-only state with the same transaction
marker—is resumed, while any unmarked tag or tag, release, correlation, or asset
mismatch fails closed without force-moving tags or replacing assets.

Every successful preparation or publication uploads
`native-release-result-<run-id>`. Its JSON returns the correlation identifier,
workflow run ID/attempt/URL/head SHA, native tag plus legacy `tag`, upstream
ref/commit, native commit, publication artifact ID/URL/digest, manifest and
checksum digests, exact checksum entries, required/actual bundle coverage,
smoke conclusion, and exact GitHub Release and asset metadata when published.
Publishing is rejected unless the required Linux x64 package-load smoke passes;
`smoke_policy=skip` is explicit preparation-only behavior.

See [Native Release Version Policy](docs/release_version_policy.md) for ordering,
prerelease classification, legacy compatibility, and the downstream contract.

## Backend Policy (Worthy Sets)

Release CI builds backend lanes separately where needed and merges them into
the following per-target bundles:

- Android: arm64 = Vulkan + OpenCL + CPU variants (Kleidi-enabled where safe); x86_64 = Vulkan + OpenCL + CPU
- iOS/macOS: Metal + CPU (consolidated into `libllamadart`, BLAS/Kleidi disabled)
- Linux x64: Vulkan + CUDA + BLAS + CPU (HIP/ROCm built in a separate release job)
- Linux arm64: Vulkan + BLAS + Kleidi + CPU
- Windows x64: Vulkan + CUDA + BLAS + CPU
- Windows arm64: Vulkan + BLAS + Kleidi + CPU

Non-Apple targets use `GGML_BACKEND_DL=ON`, so backend libs are optional at package/runtime level.

## Runtime Packaging Model

Release assets contain:

- Apple: consolidated `libllamadart` per target.
- Apple SPM: `llamadart-native-apple-xcframework-<tag>.zip`, a
  `llamadart_native.xcframework` built from the same Apple slices and wrapper
  code as the native-assets tarballs. Each framework slice embeds a
  `PrivacyInfo.xcprivacy`; the per-target Apple tarballs are bare dylibs and
  carry none. See [`docs/apple_privacy_manifest.md`](docs/apple_privacy_manifest.md).
- Non-Apple core libs: `llamadart`, `llama`, `llama-common`, `ggml`, `ggml-base` (and `mtmd` where produced)
- Non-Apple backend libs: `ggml-<backend>` modules (`ggml-vulkan`, `ggml-opencl`, etc.)
- Windows backend runtime deps:
  - CUDA lanes include CUDA runtime DLLs required by `ggml-cuda` (for example `cudart64_*.dll`, `cublas64_*.dll`).
  - BLAS lanes include `openblas*.dll` required by `ggml-blas`.
  - NVIDIA driver DLLs (for example `nvcuda.dll`) are not bundled and are provided by GPU drivers.
- Headers archive: `llamadart-native-headers-<tag>.tar.gz` with `llama_cpp/...` and `libllamadart/...` roots, including llama.cpp, ggml, mtmd, and `llama_dart_wrapper.h`.

Consumers can choose which backend libs to include in their package and load at runtime.

## Release Asset Naming

Assets are suffixed with platform/arch, for example:

- `libllamadart-linux-x64.so`
- `libllama-linux-x64.so`
- `libggml-vulkan-linux-x64.so`
- `libggml-opencl-android-arm64.so`
- `ggml-cuda-windows-x64.dll`

## Repository Layout

- `.github/workflows/auto_native_release.yml`: stable upstream candidate detector; dispatches the native release workflow for an exact unbuilt stable release and never publishes assets itself.
- `.github/workflows/native_release.yml`: build + package + release.
- `.gitmodules`: pinned native dependency submodules.
- `CMakeLists.txt` + `CMakePresets.json`: root-native build configuration.
- `src/`: `llama_dart_wrapper.*`.
- `third_party/llama.cpp`: upstream llama.cpp submodule.
- `third_party/Vulkan-Headers`: Vulkan API headers submodule for Android Vulkan builds.
- `third_party/SPIRV-Headers`: SPIR-V registry headers required by the `llama.cpp` Vulkan backend.
- `third_party/OpenCL-Headers`: OpenCL headers submodule (Android OpenCL builds).
- `third_party/OpenCL-ICD-Loader`: OpenCL loader submodule used to produce Android `libOpenCL.so` when NDK does not provide one.
- `third_party/opencl-stubs`: optional local fallback location for OpenCL headers/stubs.
- `tools/build.py`: cross-platform build entrypoint.
- `tools/validate_exports.py`: verifies required wrapper C exports, including
  exit-teardown, speculative-decoding and TTS symbols, in release artifacts.
- `tools/package_linux_artifact.py`: preserves Linux ELF version files and
  SONAME symlinks while transporting split CI artifacts.
- `tools/validate_linux_artifact.py`: checks Linux archive members, symlinks,
  SONAMEs, and local `DT_NEEDED` dependencies, that no library built here
  imports `__cxa_atexit`, and that `libllamadart.so` cannot be unloaded.
- `tools/package_apple_xcframework.py`: packages Apple `libllamadart` slices as
  an SPM-compatible XCFramework zip.
- `tools/apple/PrivacyInfo.xcprivacy`: privacy manifest embedded in every
  XCFramework slice.
- `tools/validate_apple_xcframework.py`: checks each XCFramework slice's privacy
  manifest location, contents, and declared required-reason API categories
  against the slice binary's imports.
- `scripts/generate_assets_manifest.sh`: builds `assets.json` + checksums.
- `scripts/auto_release_dispatch.py`: decides whether daily discovery authorizes
  one exact native release dispatch and emits its immutable inputs.
- `scripts/verify_release_provenance.py`: verifies exact-source checkout, source
  tag, and manifest provenance contracts.
- `docs/platform_backend_strategy.md`: platform/backend matrix.
- `docs/release_version_policy.md`: stable/nightly channels, wrapper rebuild
  ordering, provenance, and downstream coordination.
- `docs/parity_tools.md`: local build recipe and naming convention for upstream
  `llama-cli` parity checks against a released native tag.

## Local Build (Preferred)

Builds are primarily driven by root `CMakePresets.json` via `tools/build.py`.
Android arm64 CPU variants use isolated CMake build directories so per-variant
ISA flags remain correct while packaging the full variant matrix. The raw
`android-arm64-v8a-full` preset now represents the primary arm64 build, while
`tools/build.py` assembles the additional CPU variant outputs.

Examples:

```bash
# macOS arm64 (Metal + CPU)
python3 tools/build.py apple --target macos-arm64

# Linux x64 (Vulkan + CUDA + BLAS + CPU)
python3 tools/build.py linux --arch x64

# Linux x64 CPU-only artifact validation build
python3 tools/build.py linux --arch x64 --backend cpu

# Linux x64 HIP/ROCm build
python3 tools/build.py linux --arch x64 --backend hip

# Android both ABIs (arm64: Vulkan + OpenCL + CPU variants; x86_64: Vulkan + OpenCL + CPU)
python3 tools/build.py android --abi all

# Windows x64 (Vulkan + CUDA + BLAS + CPU)
python3 tools/build.py windows --arch x64

# Windows arm64 (Vulkan + BLAS + Kleidi + CPU)
python3 tools/build.py windows --arch arm64
```

List supported combinations:

```bash
python3 tools/build.py list
```

Initialize submodules after clone:

```bash
git submodule update --init --recursive
```

This build produces the shipped runtime libraries only; upstream tools such as
`llama-cli` are force-disabled here. See
[Parity Tool Builds](docs/parity_tools.md) for building them from a released
tag's pinned `llama.cpp` commit in a separate clone.

## Exit Teardown

ggml-metal aborts in its static destructor
(`GGML_ASSERT([rsets->data count] == 0)`) when the process exits while a Metal
buffer is still allocated. A Flutter macOS quit calls C `exit` without shutting
the Dart isolates down, and a hot restart drops them without running their
finalizers, so the caller cannot free its models in time.

`libllamadart` therefore tracks live objects in a registry and frees what is
left of it at exit:

- `llama_dart_model_load_from_file`, `llama_dart_init_from_model` and
  `llama_dart_mtmd_init_from_file` create a tracked object. So do the TTS,
  speculative, MTP and n-gram init functions. Tracking happens inside the
  creating call. `llama_dart_exit_track` tracks any other object.
- `llama_dart_exit_free` frees a tracked object and removes it, so nothing is
  freed twice. It takes one pointer and can be a Dart `NativeFinalizer`
  callback.
- On Apple platforms teardown runs during exit, before the first static of
  `libllamadart` is destroyed. It frees the remaining objects in stage order:
  sessions, schedulers, contexts, model users such as mtmd contexts, backends,
  then models. It first waits up to 2 s (`llama_dart_exit_set_wait_ms`) for the
  calls in flight on other threads, and frees nothing if one is still running.
  A model load in flight is cancelled.
- `llama_dart_decode`, `llama_dart_encode`, the other `llama_dart_` functions
  named after an upstream function, and the wrapper's own session functions
  are calls in flight. The decoding ones also wait for the backend, because
  Metal computes after `llama_decode` returns. Make every long native call on
  a tracked object through one of them: a Dart isolate can be killed inside any
  native call, and only a call that begins and ends in native code is then
  still accounted for.
- Once teardown has begun, a thread that reaches one of these functions outside
  a call in flight never returns from it: the objects it holds are gone. It
  stays blocked until the process is gone, so a static destructor of another
  library that joins such a thread hangs the exit.
- A native call on a tracked object that is not a call in flight is only safe
  while it is short: teardown allows 250 ms after a thread's last call in
  flight. A longer one is a use after free at exit.
- TTS, speculative and MTP state over a context that was created with the
  upstream function is tracked but not freed at exit, so a caller that has not
  switched to the tracked functions keeps its previous exit behavior.

On Linux with glibc, C `exit` destroyed the statics of llama.cpp under the
threads that were still inside it, and those crashed
([llamadart#949](https://github.com/leehack/llamadart/issues/949)). There:

- No library of the bundle registers static destructors
  (`src/llama_dart_static_destructors.c`), so their statics stay valid until
  the process is gone. On Linux they are separate libraries, and no exit
  handler of `libllamadart` can be ordered before the destructor of a static
  that another library creates later.
- C `exit` does not run teardown. With a call in flight on another thread it
  ends the process at once, as `_exit` does: with the exit status and the
  standard streams flushed, and without the exit handlers and destructors
  that would have run next, the host's among them. Those would run under the
  call, and waiting for the call is not harmless: a Dart VM aborts when its
  isolates run while `exit` takes its time.
- With no call in flight the exit goes on at once. Nothing is freed and no
  thread is blocked. Afterwards a guarded call that begins on another thread
  returns its failure value with `the process is exiting` as the last error,
  and `llama_dart_exit_free` frees nothing, so a host whose own exit handler
  joins a worker gets it back.
- The handler is registered at the first creating call and again at the first
  tracked object. An exit handler that another library registers later runs
  before it. `libllamadart.so` is linked with `-z nodelete`.

[`docs/v060_2_wrapper_rebuild.md`](docs/v060_2_wrapper_rebuild.md) has the
measurements and what is not verified.

On other platforms nothing runs at exit. `llama_dart_exit_teardown` runs
teardown on request on every platform, frees included, and must be followed
directly by `exit`. It is for native hosts: like `llama_dart_exit_call_begin`
and `llama_dart_exit_call_end`, it is not meant to be bound from Dart.
`src/llama_dart_wrapper.h` documents each function.

`llamadart_exit_teardown_test` covers the registry on every platform. On macOS
and Linux it also writes a small decoder and a small encoder model and runs
the scenarios that load one, so ctest needs no model file:

```bash
cmake -S . -B build/exit-teardown -G Ninja -DLLAMADART_BUILD_TESTS=ON
cmake --build build/exit-teardown --target llamadart_exit_teardown_test
ctest --test-dir build/exit-teardown -R exit_teardown --output-on-failure
```

`-DLLAMADART_EXIT_TEARDOWN_TEST_MODEL=/path/to/model.gguf` runs them against
another model, and adding
`-DLLAMADART_EXIT_TEARDOWN_TEST_MMPROJ=/path/to/mmproj.gguf` also runs the mtmd
scenarios. Three scenarios need models of their own and run by hand:

```bash
build/exit-teardown/llamadart_exit_teardown_test model-mtp target.gguf mtp-draft.gguf
build/exit-teardown/llamadart_exit_teardown_test model-tts qwen3-tts.gguf mmproj-qwen3-tts.gguf
build/exit-teardown/llamadart_exit_teardown_test model-idle-untracked model.gguf
```

The last one exits with a model that is not tracked, and aborts on Metal.

## Exception Barrier

llama.cpp reports some failures by throwing a C++ exception, for example
`Unexpected empty grammar stack after accepting piece` when a grammar sampler
accepts a token that its grammar rejects. An exception that crosses the C ABI
into a Dart FFI caller ends the process, so every `llama_dart_` function that
calls llama.cpp or allocates catches it, returns a failure value and keeps the
message for the calling thread:

| Return type | After a caught exception |
| --- | --- |
| pointer | `NULL` |
| `bool` | `false` |
| `int32_t` status or count | `LLAMA_DART_STATUS_EXCEPTION` (`INT32_MIN`) |
| `llama_token` (`llama_dart_sampler_sample`) | `LLAMA_TOKEN_NULL` |
| `size_t` | `0` |
| `enum ggml_status` | `GGML_STATUS_FAILED` |
| `enum llama_dart_tts_status` | `LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR`, and the task fails |
| `void` | nothing; only the message is kept |

`llama_dart_last_error` returns the message as valid UTF-8, or `NULL` when the
thread's last such call caught nothing: each of them clears it on entry, which
tells a caught exception from a failure that llama.cpp reports through the
same value. The functions that free an object only set it, since a Dart
finalizer may run one between a failed call and the read of its error.
The upstream functions that can throw and that a caller would otherwise call
directly have a wrapper too: `llama_dart_sampler_accept`,
`llama_dart_sampler_init_grammar_lazy_patterns`, `llama_dart_tokenize`,
`llama_dart_token_to_piece`, `llama_dart_memory_clear`, the three
`llama_dart_mtmd_bitmap_init_` functions and the `llama_dart_ggml_backend_`
functions that reach a backend. `docs/v060_1_wrapper_rebuild.md` classifies
every upstream function that `llamadart` calls. A failed `GGML_ASSERT` still
aborts. `src/llama_dart_wrapper.h` lists the functions without a barrier and
what a caller may do with the objects of a call that failed.

`llamadart_exit_teardown_test` covers it with the `barrier-free`,
`barrier-free-idle`, `barrier-grammar` and, on macOS, `model-barrier` and
`model-sample-accept` scenarios. The grammar ones use a model with a
vocabulary that the test writes. `llamadart_barrier_test` calls the helpers
behind the barrier directly.

## Vulkan Device Facts

`llama_dart_vulkan_get_device_count` and `llama_dart_vulkan_get_device_info`
report, for each device that ggml-vulkan registers, the Vulkan API version of
the loader and of the device, the subgroup size, the vendor and device ids,
the device type, the driver version and the name. A caller needs them before
it uses the Vulkan backend: ggml-vulkan requires Vulkan 1.2 but still
registers a device whose driver stops at 1.1, and some of its shaders depend
on the subgroup size.

They read the Vulkan loader directly (`vulkan-1.dll`, `libvulkan.so.1`, or
`libvulkan.so` on Android), which `libllamadart` opens at run time and does
not link. They create a Vulkan instance and no logical device, and do not load
ggml-vulkan. Creating the instance loads the system's GPU drivers into the
process, and a driver crash there cannot be caught, so call them only when
the Vulkan backend is being considered. Device `N` is ggml's device `VulkanN`: the selection of
`ggml_vk_instance_init` is mirrored, including `GGML_VK_VISIBLE_DEVICES`.
Without a usable loader they return a negative `llama_dart_vulkan_status`,
and on Apple platforms always `LLAMA_DART_VULKAN_STATUS_UNSUPPORTED`.
`src/llama_dart_wrapper.h` documents the fields and the two cases where the
order can differ from ggml's.

`llamadart_vulkan_device_info_test` runs the probe against a table-driven
loader, and against the machine's own loader. To require a device there:

```bash
GGML_VK_VISIBLE_DEVICES=0 LLAMADART_TEST_VULKAN_DEVICES=1 \
  build/wrapper-contract/llamadart_vulkan_device_info_test
```

## Experimental TTS Wrapper

`libllamadart` exposes a versioned, opaque C symbol contract around llama.cpp's
experimental audio-generation helpers. The wrapper currently recognizes
Qwen3-TTS projectors, exposes model capability metadata, and
provides task start/step/cancel/reset plus caller-buffered float32 mono PCM
reads after synthesis completes. The caller owns the `llama_context` and
`mtmd_context`, must create the llama context with embeddings enabled, and must
give the TTS task exclusive access to both contexts until the task reaches a
terminal state or is reset.

The upstream API does not currently expose completed PCM incrementally, so the
wrapper's step API is not a real-time audio stream. A step sees a cancel when it
starts and when its frame generation or audio output returns. Qwen3-TTS decodes
audio in 72-frame windows, inside the step that fills a window and the step that
ends speech. Install `llama_dart_tts_eval_callback` as the mtmd context's
`cb_eval` to stop that decode at its next chunk boundary, and attach a
caller-owned cancel byte with `llama_dart_tts_set_cancel_flag` to cancel from a
thread that must not touch the task. Public Dart support should remain
experimental and capability-gated until artifact and platform validation is
complete.

The local smoke target is opt-in because it requires large model artifacts:

```bash
cmake -S . -B build/tts-smoke -G Ninja \
  -DGGML_CCACHE=OFF \
  -DGGML_METAL=OFF \
  -DLLAMADART_BUILD_TESTS=ON \
  -DLLAMADART_BUILD_TTS_SMOKE=ON
cmake --build build/tts-smoke --target \
  llamadart_speculative_api_test llamadart_tts_api_test \
  llamadart_tts_eval_test llamadart_mtmd_compat_test llamadart_tts_smoke
ctest --test-dir build/tts-smoke --output-on-failure
build/tts-smoke/llamadart_tts_smoke \
  /path/to/Qwen3-TTS-model.gguf \
  /path/to/mmproj-Qwen3-TTS.gguf \
  /tmp/tts-smoke.wav \
  "Hello from Llama Dart." en \
  --gpu
```

The smoke checks capability metadata, cancellation/reset, two consecutive
syntheses, 24 kHz mono PCM metadata, finite/non-silent output, and WAV writing.
It also checks the eval callback: uncancelled PCM stays byte-identical, frame
steps stay whole, no chunk of the final decode exceeds a quarter of it, and a
cancel issued a quarter of the way into a decode step, through
`llama_dart_tts_cancel` or a cancel flag, makes that step return `CANCELLED`
within a third of the decode's duration. That holds at the end-of-speech and
frame-72 decodes even when the flag returns to zero right after the decode
breaks, and no step returns `OK` after a break.
Omit `--gpu` for a CPU-only run. An optional speaker-reference audio path may
appear before the final `--gpu` flag.

## Windows Build Notes

MSVC release builds keep interprocedural optimization enabled by default, but
`llama-common` and `mtmd` are excluded from IPO/LTCG. Upstream `llama-common` is
a large utility DLL, and current MSVC `link.exe` can access-violate while
linking it with `/LTCG`. Upstream `mtmd` enables CMake's automatic Windows
symbol export; CMake cannot generate the export definition from LTCG object
files. These target-scoped overrides keep Windows release artifacts
reproducible without changing the runtime packaging model.

To retest MSVC IPO after a compiler or upstream change, configure with:

```bash
cmake --preset windows-x64-full \
  -DLLAMADART_MSVC_LLAMA_COMMON_IPO=ON \
  -DLLAMADART_MSVC_MTMD_IPO=ON
```

## Local Linux Build With Docker Cache

Use `tools/docker_build_linux.sh` to build Linux targets in a cached Docker
image. The image is based on NVIDIA CUDA 12.8.1 and keeps heavy dependencies
(CUDA, cross toolchains, Vulkan/BLAS dev packages) in reusable layers, so repeat
builds are faster.
This Docker flow is for local development only; CI Linux jobs run on native GitHub runners.

```bash
# Linux x64 full set
./tools/docker_build_linux.sh --arch x64 --jobs 8

# Linux arm64 full set (cross-compiled in container)
./tools/docker_build_linux.sh --arch arm64 --jobs 8

# Build both Linux targets
./tools/docker_build_linux.sh --arch all --jobs 8
```

Useful flags:

- `--clean`: clean preset build directories before build
- `--rebuild-image`: force image refresh
- `--platform`: override Docker platform (default `linux/amd64`)
- `--image`: custom image tag

Outputs are written to `bin/linux/x64` and `bin/linux/arm64`.
Note: Kleidi-enabled lanes require network access to fetch upstream Kleidi sources.
Android arm64 CPU variants are built in isolated configurations so Kleidi can
stay enabled without leaking newer ISA flags into lower-tier variant binaries.

Android OpenCL override env vars (optional):

- `OPENCL_INCLUDE_DIR=/path/to/opencl/headers`
- `OPENCL_LIBRARY_ANDROID_ARM64_V8A=/path/to/arm64/libOpenCL.so`
- `OPENCL_LIBRARY_ANDROID_X86_64=/path/to/x86_64/libOpenCL.so`

## Maintainer Docs

- `AGENTS.md`: agent workflow and cross-repo handoff
- `CONTRIBUTING.md`: contributor setup/build/release steps
