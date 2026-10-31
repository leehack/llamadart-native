# v0.8.0-1 wrapper rebuild

`v0.8.0-1` is upstream v0.8.0 plus patch 0004. Metal context creation in
`v0.8.0` fails on iOS and macOS before 26 ([#154](https://github.com/leehack/llamadart-native/issues/154)).

## Patch 0004

llama.cpp #31146 made the `bfloat4` specializations of `ggml_metal_unary_cast`
`constexpr`. Older Metal compilers reject a vector subscript in a constexpr
function, and the `float4` specialization has four. Upstream #31360 drops
`constexpr` from both specializations. It is in `b11983` and later, in no stable
release.

A patch cannot edit a `.metal` file (`patches/llama.cpp/README.md`), so 0004
makes the same edit to the embedded source in `ggml-metal-device.m` as it loads
it. Its `upstream_fix` retires it when the pinned upstream has #31360.

## OS floor check

ggml embeds Metal source and compiles it on the user's device, so the OS decides
what compiles. Xcode does not stand in for it: `-std=metal3.2` fails on Xcode 16.4
and passes on Xcode 26. Neither does loading a model on a runner: a paravirtual
device reports family Apple5 and no bfloat, so ggml leaves out the failing code
there, and a Metal job on such a runner passes whatever the OS.

`metal_os_floor.yml` loads the backend from the built `libllamadart.dylib`
through ggml's own loader, as reported and with the bfloat families reported
(`tools/apple/metal_floor_check.m`), on `macos-14`, `macos-15`, `macos-26` and the
iOS 17.0 and 18.6 simulators, whose runtimes bring that iOS version's compiler.
It runs on pull requests that can change the kernels and gates
`native_release.yml`. No runner has an iOS 16 runtime, and Firebase's iOS 16.6
device did not start the XCTest bundle with either runtime.

## Evidence

Firebase, the 0.17.1 XCTest bundle with only `llama.framework/llama` replaced by
a v0.8.0 plus 0004 build: iPhone 16 Pro iOS 18.3 and iPhone SE 3 iOS 18.4 go from
`failed to build 'unary' library` to `loaded 36 libraries`, and the cases that
pass on Metal in core 0.17.0 pass. `C06.history` fails on capitalization in
all of these runs, 0.17.0 and CPU included. The probe bundle swapped the same way
loads `GpuBackend.auto` on Metal with 999 GPU layers on both.
