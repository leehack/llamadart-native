# v0.6.0-2 rebuild

Changes on llama.cpp `v0.6.0`
(`d81235049384534c167caea52b85a694f6103d14`) for the native rebuild tag
`v0.6.0-2`. The exported header is unchanged from `v0.6.0-1`.

## Carried llama.cpp patches

This is the first release built with patches on top of the pinned upstream
source. `patches/llama.cpp/README.md` has the conditions, the file format and
the procedure. In short:

- `third_party/llama.cpp` is not modified. While CMake configures,
  `tools/llama_cpp_patches.py` writes a patched copy of each source that a
  patch edits to `<build>/llama.cpp-patched/`, and
  `cmake/llama_cpp_patches.cmake` makes the upstream target compile the copy.
  `cmake/kleidiai_windows.cmake` and the Android CPU backend score object
  already add sources to upstream targets from this repository's CMake; this
  replaces one.
- A hunk applies only where its context and removed lines occur exactly once.
  A patch that does not fit the llama.cpp being built is skipped with a CMake
  warning, because the qualification lanes build older upstream commits:
  patch 0001 does not fit the `post-v0.4.0` commit.
- `tools/validate_android_artifacts.py` fails an Android bundle in which a
  library lacks the text that a patch compiles into it. The release runs it,
  so a release, the automatic stable one included, cannot lose a patch that
  upstream has outgrown: it fails until the patch is refreshed or deleted.
- `assets.json` gains `llama_cpp_patches`, the `file`, `sha256` and `tracking`
  issue of each patch, and `[]` for a release without any.

A consumer that compares against upstream `llama.cpp` at `llama_cpp_commit`
(`docs/parity_tools.md`) compares against unpatched source. The patches of
this rebuild change only `ggml-vulkan`, each for the condition its section
names.

## Adreno 750 shader compiler crash

Issue: [llamadart-native#79](https://github.com/leehack/llamadart-native/issues/79).
Patch: `0001-vulkan-adreno-750-matvec-shared-memory-reduction.patch`.

On a Galaxy S24 (`SC-51E`, Adreno 750, driver version `2150604839`, build
`c81b28e1a5`) the first prompt decode on Vulkan ends the process: `SIGSEGV` at
`0x2d0` in the driver's shader compiler (`CreateQGLCProgram`), under
`vkCreateComputePipelines`. A crash inside the driver cannot be caught.

Root cause at `d81235049`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
`ggml_vk_load_shaders` sets `use_subgroups = device->subgroup_arithmetic`
(line 2899) and with it builds every dequantize matrix-vector pipeline in the
subgroup or hybrid reduction mode (lines 2927 to 2929), requiring full
subgroups of the device's subgroup size, 64 here (lines 2910 and 2936;
`eRequireFullSubgroupsEXT` and `requiredSubgroupSize` at lines 731 and 742).
The pipelines are compiled on first use, from `ggml_vk_mul_mat_vec_q_f16`
(line 6638). This driver's compiler crashes on those shaders once the batch
has enough columns: 5 for F32, 3 for F16, BF16, Q4_0, Q4_1, Q5_0 and Q5_1, 2
for IQ4_NL. A single-token decode compiles, so the crash comes with the first
prompt. Upstream `master` at `609290be6` (2026-10-09) has the same code.

The patch sets `use_subgroups = false` for that block when the device is
exactly this driver (vendor `0x5143`, `VK_DRIVER_ID_QUALCOMM_PROPRIETARY`,
driver version `2150604839`, name `Adreno (TM) 750`), in Android builds only.
The pipelines then use the shared-memory reduction and no required subgroup
size, which is what ggml already does for a device without subgroup
arithmetic and, for part of the block, for the Imagination driver. It is the
source change of the draft
[ggml-org/llama.cpp#29165](https://github.com/ggml-org/llama.cpp/pull/29165)
(`8640bb336`), without its build option.

Evidence, from the issue: Galaxy S24 `SC-51E`, Android 16, Firebase Test Lab,
a JNI app on the llama.cpp C API, context 1024, batch 1024, micro-batch 512,
flash attention off, every layer on the GPU, greedy decoding, 32 tokens,
compared with the CPU in the same app. llama.cpp was `b11256` (`c85b92c69`),
not `v0.6.0`.

| Build | stories15M F32 | Qwen3.5-0.8B Q4_0 |
| --- | --- | --- |
| `b11256` | crash | crash |
| `b11256` with this change | 32 of 32 tokens equal to CPU | 32 of 32 tokens equal to CPU |

What the patch does not fix, on this driver:

- Matrix-vector products of Q8_0, Q2_K to Q6_K, IQ4_XS and IQ3_S weights give
  wrong results (relative error 0.06 to 0.31) with and without the patch, and
  Q4_K with one column fails pipeline creation with `VK_ERROR_UNKNOWN`. A
  model in one of these types, which includes the common `Q4_K_M`, does not
  work on Vulkan on this driver.
- Batches above 8 columns, flash attention, `mul_mat_vec_id` and F16
  activations were not run on the device.
- Another driver version or another Adreno is not matched and behaves as
  before.

Remove the patch when the pinned upstream has an equivalent change, or when
the S24 run passes without it.

## Null `vkGetBufferDeviceAddress`

Issue: [llamadart#782](https://github.com/leehack/llamadart/issues/782).
Patch: `0002-vulkan-buffer-device-address-entry-point.patch`.

On a Galaxy A53 5G (`SC-53C`, Mali-G68, Android 16) loading a model on Vulkan
ends the process: `SIGSEGV` at `pc 0` called from `libggml-vulkan.so`, under
`ggml_backend_alloc_ctx_tensors_from_buft` and
`llama_model_base::load_tensors`, with native `v0.5.0-2`.

Root cause at `d81235049`, `ggml/src/ggml-vulkan/`:

- `ggml_vk_instance_init` requires Vulkan 1.2 only of the loader
  (`vk::enumerateInstanceVersion`, `ggml-vulkan.cpp` lines 5192 to 5197).
  Nothing compares the `apiVersion` of the device.
- `ggml_vk_get_device` declares `VkPhysicalDeviceVulkan11Features` and
  `VkPhysicalDeviceVulkan12Features` without initializing their members (lines
  4308 to 4316), passes them to `vkGetPhysicalDeviceFeatures2` (line 4475) and
  takes `buffer_device_address` from `vk12_features.bufferDeviceAddress` (line
  4518). A driver below Vulkan 1.2 does not know these structures and writes
  nothing, so the value is what the stack held.
- `ggml_vk_create_buffer` then calls `getBufferAddress`, the core 1.2
  `vkGetBufferDeviceAddress` (`ggml-vulkan-buffers.cpp` lines 175 to 177),
  through a Vulkan-Hpp dispatcher that is initialized from the instance (lines
  5190 and 5270 of `ggml-vulkan.cpp`) and never from the device. A device
  without the function makes that a call of a null pointer.

Upstream `master` at `609290be6` (2026-10-09) has the same code, and no
upstream issue or pull request names it.

The patch asks the created device for the function
(`vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress")`, after
`createDevice` at line 4769) and clears `buffer_device_address`, with a
warning, when there is none. ggml then creates buffers without
`eShaderDeviceAddress` and uses the im2col shaders without device addresses,
as on any device that lacks the feature.

Not known: whether the A53's driver reports itself below Vulkan 1.2. The run
that crashed recorded no Vulkan version. A driver at 1.2 or later that
reports the feature without the function would crash the same way, and the
patch covers both because it tests the function and not the version.

What the patch does not fix: on a device below Vulkan 1.2 the other members
read from the two structures stay indeterminate (`shaderFloat16`,
`storageBuffer16BitAccess`, `vulkanMemoryModel`), so ggml-vulkan still needs
Vulkan 1.2 from the driver. `llamadart` refuses such a device before a load
from the `api_version` of the `v0.6.0-1` device facts, and then never reaches
this code. The patch is for a driver at 1.2 or later without the function and
for a caller that does not check.

No device has run the patch. That the device answers null follows from the
trace and from the issue's note that initializing the dispatcher from the
device did not help in stable-diffusion.cpp's copy of this code.

Remove the patch when the pinned upstream checks the device's API version or
the function before it uses buffer device addresses.

## Device runs owed before release

No lane runs ggml-vulkan on a device, and both patches change behavior only on
the affected drivers. Before `llamadart` ships with `v0.6.0-2`, on the bundle
of that tag or of a non-publishing run of the release workflow, Vulkan load
and generation compared with CPU output on the same device:

| Device | Shows |
| --- | --- |
| Galaxy S24 `SC-51E`, Adreno 750, API 36 | stories15M F32 and Qwen3.5-0.8B Q4_0 generate without the crash and equal CPU, this time on llama.cpp `v0.6.0` and through `llamadart` |
| Galaxy A53 `SC-53C`, Mali-G68, API 36 | the device's Vulkan API version; then a typed refusal or CPU fallback from `llamadart` (below 1.2), or Vulkan output equal to CPU with the warning of patch 0002 in the log (1.2 or later) |
| Pixel 9 Pro `caiman`, Mali-G715 ([llamadart#948](https://github.com/leehack/llamadart/issues/948)) | no change from `v0.6.0-1`: neither patch applies to this driver |

## Checks

```bash
git submodule update --init --recursive
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_release_provenance.py
cmake -S . -B build/v060 -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_METAL=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/v060 --parallel 8
ctest --test-dir build/v060 --output-on-failure
```

`tests/test_llama_cpp_patches.py` covers the patch tool on its own sources and,
with the submodule checked out, requires the carried series to fit it. The
`android-vulkan-shaders` lane of `validate_wrapper.yml` compiles the patched
`ggml-vulkan.cpp` with the release NDK against the pinned, `v0.5.0` and
`v0.6.0` upstreams, which both patches fit, and on the pinned one validates
the bundle, markers included. No lane runs ggml-vulkan on a device.

## Linux host shutdown

### Contract

Before a native host calls C `exit()`, it stops new work, stops and joins its
native workers, and shuts down its Dart isolates or Flutter engine. A kill
request is not completion: await each isolate's exit notification and native
finalizers before exiting. Free contexts before projectors / models. A native
call already in progress completes before its isolate can shut down.

Linux, Android excepted, retains the statics owned by the runtime bundle.
`src/llama_dart_static_destructors.c` supplies a hidden `__cxa_atexit` that drops
owned static destructors; external libraries keep their ordinary exit behavior.
The wrapper remains linked with `-z nodelete` so held objects and finalizer
callbacks remain valid across `dlclose`. Artifact validators reject missing
retention and owned imports of `__cxa_atexit`.

C exit registers no automatic Linux wrapper teardown or exit handler. It does
not set a late-call refusal state, flush streams early, wait, free live objects
or call `_exit`. Normal host callbacks, dependency destructors and libc stdio
flushing run in their normal order. An earlier native-host callback can still
join a worker and free its tracked objects; the wrapper does not park that
worker merely because libc exit began.

Apple automatic teardown and explicit `llama_dart_exit_teardown` are unchanged.
Do not invoke explicit teardown before joining workers: it parks later guarded
calls, so a subsequent host join can hang. Android is unchanged. Windows / musl
hardware qualification is not established by these Linux glibc measurements.

### Why exit must follow host shutdown

A direct C exit with live Dart isolates is unsafe independently of llama.cpp.
With an allocating isolate and an earlier300ms host callback, the VM aborts at
`handles_impl.h:39` without loading any native model (llamadart#977). Retaining
owned statics cannot protect a GPU driver's or BLAS library's destructors either.
Direct C exit while Dart isolates or dependency-using native workers are alive
is outside this contract, including calls from an evaluation callback or an
incomplete model load. CPU raw-exit controls remain diagnostics, not support
claims. A hung native call cannot be called quiescent after a timeout; the host
must withhold normal C exit until it completes or explicitly choose its own
abrupt process-termination policy.

The rejected candidate flushed streams and called `_exit` when a call was in
flight. It skipped earlier host callbacks, and its `fflush(NULL)` deadlocked if
a worker held a FILE lock whose earlier host callback would release it. Removing
only the flush lost buffered output. Both alternatives violate the accepted
preservation contract and are absent from this revision.

### Validation and replay

`llamadart_exit_teardown_exit-status` verifies exit37, the earlier callback,
buffered output, and a host callback joining a marked native worker that holds a
FILE lock. Its10s CTest timeout makes the old unconditional-flush deadlock fail.
`exit-late-calls` verifies the host callback can create and free tracked objects
without an automatic Linux refusal state. These are native compatibility
controls, without a running Dart VM. Apple teardown tests retain their contract.

The maintained manual `tests/manual/dart_host_shutdown_probe.dart` uses worker
NativeFinalizers, per-worker onExit notifications, an allocating isolate, a
slow native call and a slow earlier host callback. Build the adjacent shim as
in `dart_exit_probe.dart`; set `PROBE_STEP_DELAY_MS=300`, then run:

```bash
dart tests/manual/dart_host_shutdown_probe.dart libshim.so <bundle-dir> \
  cooperative-finalizer-slow-native <model.gguf>
```

Success requires exit0, `NATIVE_ACTIVE_BEFORE_KILL`, `NATIVE_CALLS_AFTER_EXIT 0`,
`NATIVE_FREE_COUNT_AFTER_EXIT 1`, `HOST_HANDLER_COMPLETED`, and `C_BUFFERED_PAYLOAD`.
`raw-control` reproduces the VM defect; `cooperative-control` verifies allocator
shutdown without loading a native bundle. Neither case requires model downloads.

Pre-implementation prototype evidence (not qualification of this revised head):
Linuxarm64 GCC14/glibc2.41, Dart3.13.1, stories15M CPU, five repeats per case /
build, released baseline vs rejected head vs preservation prototype.60/60
cooperative model cases and15/15 no-model cases retained callbacks / output.
30/30 active native-call cases waited314–339ms before onExit;15/15 finalizers
completed first.15/15 raw no-model controls aborted. The protocol also passed
on the released baseline, so it establishes the lifecycle boundary, not a new
native feature. The GCC13 dependencies were reused read-only; this differs
from Ubuntu24.04/GCC13 release qualification.

Current-head CI, independent audit and exact-runtime replay are required anew.
Remaining release gates include real GPU / OpenBLAS driver shutdown, image
runtime, Flutter Linux host ordering, final published artifacts and device
qualification. No earlier raw-exit result establishes those gates.
