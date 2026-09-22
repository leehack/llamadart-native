# Parity Tool Builds

Native releases publish runtime bundles, an Apple XCFramework, and headers.
They do not publish standalone upstream tool binaries such as `llama-cli`.
A throughput or correctness investigation that wants to compare `llamadart`
against upstream therefore has to build the tool itself, and the comparison is
only meaningful when that tool is built from the exact `llama.cpp` commit the
consumed native release was built from.

This document is the supported local build recipe and artifact naming
convention for those parity checks. Only macOS arm64 Metal is verified; see
[Verified coverage](#verified-coverage).

## 1. Read the pinned upstream commit

`assets.json` records the release tag in `tag`. Releases built under the
current contract also record the requested upstream ref (`llama_cpp_tag`), its
resolved full commit SHA (`llama_cpp_commit`), and the release tag again in
`native_release_tag`; `scripts/release_contract.py:347-358` enforces that those
fields match the dispatch inputs the release was built from, and that
`native_release_tag` and `tag` are equal. Where `llama_cpp_commit` is present it
is the authoritative upstream provenance — not the `llama.cpp` submodule pointer
in whatever `llamadart-native` checkout you have locally.

Older releases predate those fields and carry only `tag`. Read `tag` and treat
the upstream fields as optional; a command that indexes them directly fails on
exactly the releases that need the fallback below.

```console
$ GH_HOST=github.com gh release download v0.4.1 --repo leehack/llamadart-native \
    --pattern assets.json --dir /tmp/llamadart-native-v0.4.1
$ python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["tag"], d.get("llama_cpp_tag","<absent>"), d.get("llama_cpp_commit","<absent>"))' \
    /tmp/llamadart-native-v0.4.1/assets.json
v0.4.1 v0.4.1 b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
```

When `llama_cpp_commit` is absent the manifest does not pin the upstream source
at all. `b9873`, the release the
[leehack/llamadart#274](https://github.com/leehack/llamadart/issues/274) run
consumed, is one of those:

```console
$ GH_HOST=github.com gh release download b9873 --repo leehack/llamadart-native \
    --pattern assets.json --dir /tmp/llamadart-native-b9873
$ python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["tag"], d.get("llama_cpp_tag","<absent>"), d.get("llama_cpp_commit","<absent>"))' \
    /tmp/llamadart-native-b9873/assets.json
b9873 <absent> <absent>
$ git ls-remote --tags https://github.com/ggml-org/llama.cpp.git refs/tags/b9873
a4107133a634250c8c9d888bc0bc8520dcfd6105	refs/tags/b9873
```

The native tag's prefix is the upstream ref by policy
(`docs/release_version_policy.md`), so resolving that ref at `ggml-org/llama.cpp`
recovers a commit. Record that the commit came from a tag lookup rather than
from the manifest: a tag can be moved, a recorded SHA cannot.

Use the native tag your `llamadart` checkout actually consumes, not the latest
release.

## 2. Clone upstream separately

Check the commit out in its own clone. Do not run `git checkout` inside
`third_party/llama.cpp`: that moves the submodule pointer and leaves the
wrapper worktree dirty, which silently changes what a subsequent wrapper build
produces.

A depth-1 fetch of the single commit is enough.

```console
$ mkdir -p /tmp/upstream-llama.cpp && cd /tmp/upstream-llama.cpp
$ git init -q
$ git remote add origin https://github.com/ggml-org/llama.cpp.git
$ git fetch --depth 1 origin b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
From https://github.com/ggml-org/llama.cpp
 * branch            b29c606e28a01b1bc8c1351026a0fa6e616bf6c4 -> FETCH_HEAD
$ git checkout -q FETCH_HEAD
$ git rev-parse HEAD
b29c606e28a01b1bc8c1351026a0fa6e616bf6c4
```

## 3. Configure

Three groups of cache variables matter, and they have different reasons.

**Tool gating.** Upstream's `tools/CMakeLists.txt` puts `tools/cli` — the
directory that defines the `llama-cli` target — inside `if (LLAMA_BUILD_SERVER)`,
and `llama-cli` links `llama-cli-impl`, which links `llama-server-impl`. At
`b29c606e`, `LLAMA_BUILD_TOOLS=ON` alone is not enough. Configuring with
`LLAMA_BUILD_TOOLS=ON` and `LLAMA_BUILD_SERVER=OFF` succeeds, and then:

```console
$ cmake --build build/parity-probe-serveroff --target llama-cli -j 6
ninja: error: unknown target 'llama-cli', did you mean 'llama-app'?
```

So both `LLAMA_BUILD_TOOLS=ON` and `LLAMA_BUILD_SERVER=ON` are required, and
the server's HTTP layer is a transitive dependency of `llama-cli` rather than
something to opt out of.

**Host and network isolation.** Turning the server on exposes two upstream
options that default to `ON` and put content from outside the pinned commit
into the binary. `LLAMA_OPENSSL` (upstream `CMakeLists.txt:144`) makes the
build search for OpenSSL and link it when found, which on macOS means the
host's Homebrew `libssl`/`libcrypto` by absolute path; this repository's root
`CMakeLists.txt:86` forces it off for the release build.
`LLAMA_USE_PREBUILT_UI` (upstream `CMakeLists.txt:138`) downloads WebUI assets
from Hugging Face during the build, and a depth-1 clone reports build number 1,
so the pinned `b1` path misses and the download falls back to `latest`. Set both
to `OFF`. With the UI off the build embeds zero assets and prints one warning
(shown in step 4); the embedded UI is inert for `llama-cli` either way.

This is also why the recipe needs a separate clone rather than the wrapper
tree: root `CMakeLists.txt` forces `LLAMA_BUILD_TOOLS=OFF` and
`LLAMA_BUILD_SERVER=OFF` (`CMakeLists.txt:83-84`) before adding the llama.cpp
subdirectory, so no combination of `tools/build.py` arguments can produce these
binaries. Do not copy `tools/build.py`'s cache variables across wholesale; the
wrapper's overrides exist to shape a shipped runtime library, not a tool.

**Compute parity.** Mirror `GGML_NATIVE=OFF`. Release artifacts are built with
it off (`CMakeLists.txt:87`, and `base` in `CMakePresets.json`), so the shipped
runtime is never compiled for instructions found by probing the build host.
Upstream defaults it to `ON` for a non-cross build with `SOURCE_DATE_EPOCH`
unset (upstream `ggml/CMakeLists.txt:105-110`), so a tool left at the default is
compiled for the host it happens to be built on, and its throughput is not
comparable. The remaining flags below mirror `macos-arm64-full` and its
`macos-base` / `apple-base` / `base` ancestors in `CMakePresets.json`.

One parity difference the flags do not remove: `apple-base` sets
`LLAMADART_CONSOLIDATE=ON`, and root `CMakeLists.txt:62-65` then forces
`BUILD_SHARED_LIBS=OFF` and `GGML_BACKEND_DL=OFF`, so the shipped macOS runtime
is a consolidated static build, while this recipe links the same sources as
shared libraries resolved through `@rpath`. Whether that moves a given
measurement has not been established here; record it next to the number.

The binary directory carries the artifact name from
[Artifact naming](#6-artifact-naming), because the build tree is where the
tool has to stay.

```console
$ cmake -S . -B build/llama-tools-v0.4.1-macos-arm64-metal -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3 \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_METAL_USE_BF16=OFF \
    -DGGML_BLAS=OFF \
    -DGGML_CPU_KLEIDIAI=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF
...
-- Configuring done (2.1s)
-- Generating done (0.1s)
```

## 4. Build

```console
$ cmake --build build/llama-tools-v0.4.1-macos-arm64-metal --target llama-cli -j 6
...
[135/268] Provisioning UI assets
CMake Warning at <...>/scripts/ui-assets.cmake:584 (message):
  UI: no assets available - building without an embedded UI. ...
...
[266/268] Linking CXX shared library bin/libllama-server-impl.dylib
[267/268] Linking CXX shared library bin/libllama-cli-impl.dylib
[268/268] Linking CXX executable bin/llama-cli
```

It is the expected effect of `LLAMA_USE_PREBUILT_UI=OFF`, and in the verified
run it was the only warning the build emitted.

## 5. Confirm the commit before citing a number

`llama-cli --version` prints the short commit. It must match the
`llama_cpp_commit` from step 1.

```console
$ ./build/llama-tools-v0.4.1-macos-arm64-metal/bin/llama-cli --version
version: 0.4.1-dev (build 1, commit b29c606)
built with AppleClang 21.0.0.21000334 for Darwin arm64
```

Only the commit identifies the source. The `-dev` suffix is upstream's
`LLAMA_BUILD_IS_DEV` default, and `build 1` is `git rev-list --count HEAD`
(`cmake/build-info.cmake`), which the depth-1 fetch reduces to 1. The native
release build forces `LLAMA_BUILD_NUMBER`/`LLAMA_INSTALL_VERSION` to
`0`/`0.0.0` (`CMakeLists.txt:73-74`), so neither number is a parity signal on
either side.

## 6. Artifact naming

Name the CMake binary directory
`llama-tools-<native_tag>-<platform>-<arch>-<backend>`, using the same
platform/arch tokens as the release assets (`macos`, `linux`, `windows`,
`android`, `ios`; `arm64`, `x64`, `x86_64`). The verified build above is
`build/llama-tools-v0.4.1-macos-arm64-metal`.

The build tree is where the name has to go, because the tool is not
relocatable. `llama-cli` resolves `libllama-cli-impl`, `libllama-server-impl`,
`libllama`, `libllama-common`, `libmtmd`, `libggml`, `libggml-base`,
`libggml-cpu`, and `libggml-metal` through `@rpath`, and its only `LC_RPATH`
entry is the absolute path of that build directory's `bin`. Copying `bin/`
elsewhere and removing the original fails at launch (paths abbreviated):

```console
$ ./llama-cli --version
dyld[11811]: Library not loaded: @rpath/libllama-cli-impl.dylib
  Referenced from: <...> /tmp/copy-of-bin/llama-cli
  Reason: tried: '<...>/build/llama-tools-v0.4.1-macos-arm64-metal/bin/libllama-cli-impl.dylib' (no such file), ...
```

Run the tool in place, and keep the build directory for as long as the numbers
it produced are being cited.

An investigation citing a measurement should record the native release tag, the
`llama_cpp_commit`, and this directory name together. That is what the
`llamadart` speculative-decoding work in
[leehack/llamadart#274](https://github.com/leehack/llamadart/issues/274) could
not do, because it had to fall back to a nearby local build from a different
upstream line.

## Verified coverage

| Target | Status |
| --- | --- |
| macOS arm64 Metal | Verified on 2026-09-21: macOS 26.6.2, CMake 4.4.3, Ninja, AppleClang 21.0.0.21000334, upstream `b29c606e` (native `v0.4.1`) |
| macOS x86_64, iOS, Linux x64/arm64, Windows x64/arm64, Android | Unverified |

For an unverified target, take the backend cache variables from that target's
preset in `CMakePresets.json` together with everything that preset inherits —
`base` carries `BUILD_SHARED_LIBS` and `GGML_BACKEND_DL`, which the leaf presets
do not repeat, and root `CMakeLists.txt:62-70` forces both regardless of the
preset value. Keep `GGML_NATIVE=OFF` and the gating and isolation flags from
step 3, and treat the result as a starting point rather than a recipe this
repository has executed. Cross-compiled targets additionally need a runnable
host, which is not a concern for the macOS recipe above.

`llama-server` is not covered here. It is configured by the same step — that is
what `LLAMA_BUILD_SERVER=ON` turns on — but building and validating it for
parity has not been done, so its target is left unbuilt above.
