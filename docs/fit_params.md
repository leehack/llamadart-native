# GPU memory fit

The wrapper adds `llama_dart_fit_api_version()` (returns 1) and
`llama_dart_fit_params()` for [llamadart#2034](https://github.com/leehack/llamadart/issues/2034).
llama.cpp's `llama_params_fit` left `llama.h` in April 2026 and is now the C++
function `common_fit_params` in `llama-common`, in v0.8.0, v0.9.0 and v1.0.0.
The wrapper gives bindings a C entry point with an exception barrier, the exit
teardown guard and a per-device memory report.

## Contract

`llama_dart_fit_params(path, mparams, cparams, tensor_split,
tensor_buft_overrides, margins, n_ctx_min, report)` returns
`LLAMA_DART_FIT_STATUS_SUCCESS`, `_FAILURE` or `_ERROR`, or
`LLAMA_DART_STATUS_EXCEPTION` for a null argument or a caught exception.
The header has the full contract. What a binding must know:

- It measures with `no_alloc` loads, so it needs no GPU memory but reads the
  model file several times (0.1-0.4 s for a 2 GB model on an SSD).
- Only an unchanged `n_gpu_layers` (-1) is fitted, and `n_ctx` only when it is
  0. A preset layer count is refused (`FAILURE`) only when the load does not fit
  with it; when it fits, the call succeeds and changes nothing.
- "Everything fits" leaves `n_gpu_layers` at -1. "Nothing fits" is `SUCCESS`
  with `n_gpu_layers == 0`, not a failure.
- With no GPU device the call succeeds when the model fits in host memory and
  reports `n_devices == 0`, but it is not a no-op: a context of 0 comes back as
  the trained context size (65536 for the test model), and a margin larger than
  total memory is a `FAILURE`, because `margins[0]` is applied to host memory.
- `n_devices` counts every non-CPU device of the model, also one that holds
  nothing at the fitted parameters (the nothing-fits case has `n_devices >= 1`
  and `model_bytes == 0`). The `host` entry has the total and free memory of the
  CPU device, and the host's share of the weights, context and compute.
- When the fit changed the load, `mparams` points into the caller's
  `tensor_split` and `tensor_buft_overrides`. Keep them alive until the model
  is loaded. When the load fits as it is, `mparams` keeps the pointers it had.
  The override patterns are static storage inside `llama-common`: never free
  them.
- `mparams`, `cparams`, and `report` are written only on `SUCCESS`.
- The report is measured at the fitted parameters. It holds at most
  `LLAMA_DART_FIT_MAX_DEVICES` devices.
- On Windows the free memory of a Vulkan or CUDA GPU counts other processes (next
  section), and a fit whose adapter memory cannot be read fails with
  `LLAMA_DART_STATUS_EXCEPTION` instead of running on the driver's number.
- The fit estimates device memory. A later Windows context allocation or decode
  can still fail under system memory pressure
  ([llamadart-native#211](https://github.com/leehack/llamadart-native/issues/211)).

## Free memory on Windows

ggml-cuda reports `cudaMemGetInfo`, which on a WDDM adapter is the same
per-process budget (NVIDIA documents that it returns what the Windows video
memory manager says): on the L4 it was 21,663.5 MiB for `CUDA0` and for raw
`Vulkan0` while another process held 16.3 GiB.

ggml-vulkan computes free memory as `heapBudget - heapUsage` from
`VK_EXT_memory_budget`. Both are per process, and on WDDM the budget comes from
the OS video memory manager, which does not deduct memory that other processes
hold: on an L4 it stayed at 21,827.5 MiB while a process held 16,332 MiB, so the
fit chose a load that did not fit
([llamadart#2034](https://github.com/leehack/llamadart/issues/2034)).

The wrapper therefore reads the display adapter's own numbers, for every
process, with `D3DKMTQueryStatistics` (resolved from `gdi32.dll` at run time,
no static import) and the Vulkan device's LUID. For a discrete GPU it sums
`BytesResident` over the segments that are not aperture; the capacity is the
smaller of the driver's total and the sum of `CommitLimit`. Free memory is
`capacity - resident`, not the driver-reserved memory `nvidia-smi` also counts:
it matched NVML within a 4–38 MiB free difference and about 400–434 MiB used
difference in the measured samples under CUDA, D3D12 and Vulkan holders. An
integrated GPU counts all segments.

A CUDA device has no LUID in ggml's properties, so the wrapper takes it from
`cuDeviceGetLuid` of the already loaded `nvcuda.dll` (`GetModuleHandleW` and
`GetProcAddress`, never `LoadLibrary`; no CUDA link). It calls `cuInit(0)`
first, which is idempotent, and uses the ggml index as the driver ordinal after
checking that `cuDeviceGetCount` equals the registry's device count and that
`cuDeviceGetName` equals the device's description. The CUDA LUID is the DXGI
and Vulkan adapter's, so the segments and arithmetic above apply unchanged and
`CUDA0` and `Vulkan0` of one card report the same free memory.

The effective free memory is the smaller of the driver's number and the
adapter's. `common_fit_params` takes no free-memory input but uses a device's
free memory only as `free - margin`, so the wrapper adds `driver_free -
effective_free` to that device's entry of `margins` before the fit, and the
report's `free_bytes`, `llama_dart_ggml_backend_dev_memory` and
`llama_dart_ggml_backend_dev_get_props` return the effective number. The code
is in `src/llama_dart_device_memory.cpp`, with the rules in
`src/llama_dart_device_memory_internal.h` and the system call in
`src/llama_dart_d3dkmt_internal.h`.

- **The fit never runs blind.** A Vulkan or CUDA GPU or integrated GPU of the
  model whose adapter cannot be read (no LUID, a probe that does not list the
  same device, `nvcuda.dll` not loaded, a `cuInit` or `cuDeviceGet*` error,
  `D3DKMTQueryStatistics` missing or failing) fails the fit with an
  exception that names the device and the cause. The two device queries above
  fall back to the driver's number, because a listing has no fit to refuse.
- The fit reads the model's device list with one more `vocab_only` load of the
  same file before it measures; it loads no weights. That extra vocabulary-only
  load emits its normal model-metadata `INFO` lines to the active logger.
- Refused, not corrected: any `GGML_CUDA_DEVICES` count other than the physical
  count (driver and registry counts differ, including virtual devices) and a
  CUDA device with no WDDM adapter (TCC mode, no LUID). Their fit fails with the
  device named.
- Not covered: APIs other than Vulkan and CUDA, and a driver that is not WDDM.
- The fit's early return when the host has nothing to correct looks at the
  ggml registry (any `Vulkan<i>` or `CUDA<i>` GPU), not at the Vulkan probe, so
  a CUDA host without a usable Vulkan loader is still corrected. That registry
  skip is distinct from failing to resolve a selected adapter once a model
  chooses a Vulkan or CUDA GPU.
- The correction follows the model's selected device list. A registered CUDA
  backend adds no correction to a Vulkan-only or CPU-only fit.
- Memory that idle applications hold counts as used, although WDDM could take
  it back.
- `llama_dart_fit_api_version()` stays 1. A `v1.0.0` runtime has the fit but not
  this correction; bindings that need it check the release tag.

## Layer counts on v1.0.0

The loader counts the NextN/MTP layers of a model in `n_gpu_layers`;
`llama_model_n_layer`, and so `report.n_layers`, does not. v1.0.0 adds the
number of those layers to every `n_gpu_layers` that a fit sets. The report has
the difference in `n_layers_skipped` (always 0 on v0.8.0 and v0.9.0, and for a
model without NextN layers). Read the fit's result on the loader's scale:

- all layers are offloaded exactly when
  `n_gpu_layers >= n_layers + n_layers_skipped + 1` (the `+ 1` is the output
  layer), or when the fit left `n_gpu_layers` at -1;
- the fit kept layers on the CPU when `0 < n_gpu_layers` is below that bound.

Comparing `n_gpu_layers` with `n_layers + 1` alone reports a fit that kept one
layer back as a full offload.

## Concurrency

`common_fit_params` swaps the global llama log callback for every measurement
and is documented as not thread safe. The wrapper serializes fits and
`llama_dart_set_log_level` on one mutex, which also orders the lazy writes to
the override patterns. A log callback must not call
`llama_dart_set_log_level`: it runs on the fitting thread under that mutex and
would deadlock. llamadart's own callback does not.

Upstream's scope kept the original logger in its stack frame and demoted
INFO and WARN lines of every thread while it was installed. The carried patch
[0007](../patches/llama.cpp/0007-fit-log-scope-static-state.patch) keeps that
state in static storage and demotes only the lines of the thread that holds the
scope, so other threads, such as a model load in another isolate, reach the
original callback with their own levels, before, during and after a fit.
It applies to the pinned upstream qualified by native CI.

What remains: `llama_log_set` writes the callback and its user data
separately. A thread that logs in the instant a fit installs or restores the
logger can call the fit's callback with the previous user data (the fit's
callback ignores it) or the original callback with a null user data, because
the fit installs its callback with null user data. The second pair is harmless
only if the original user data is null, as llamadart's is. A thread that calls
`llama_log_set` itself during a fit has its logger overwritten when the fit
restores the original. The patch does not remove the data race in
`llama_log_set`, and TSan reports it. Installing the fit's callback while it is
already installed, which the fit lock prevents, aborts at once.

Teardown cannot cancel a fit, because its `no_alloc` loads report no progress.
It waits for a running fit, up to the exit wait limit.

## Upstream compatibility

CMake probes `LLAMADART_FIT_HAS_PATH_LIST` (v1.0.0 passes a list of paths) and
`LLAMADART_FIT_HAS_NGL_SKIP` (v1.0.0 reports skipped NextN layers). Native CI
qualifies the pinned upstream shipped by the current llamadart core minor.

## ABI delta for the Dart bindings

New exports `llama_dart_fit_api_version` and `llama_dart_fit_params`, new types
`llama_dart_fit_status`, `llama_dart_fit_device` and `llama_dart_fit_report`
(`LLAMA_DART_FIT_MAX_DEVICES` is 16). `llama_dart_fit_device` is 104 bytes.
`llama_dart_fit_report` is 1784 bytes: `n_devices`, `n_layers`, `n_ctx_train`
and `n_layers_skipped` as four `uint32_t` at offsets 0 to 12, `devices` at 16
and `host` at 1680. Nothing existing changes. The required
export list in `tools/validate_exports.py` grows from 94 to 96.

## Tests

`llamadart_fit_api_test` writes its own 32.5 MiB model. It checks the contract
(null arguments, missing and unreadable models, a failed fit that must not
change `n_ctx`) everywhere, fits that overlap each other while a sentinel
logger is installed (the logger must be the same one afterwards), and a thread
that loads a model while another fits (it must keep its INFO lines; this needs
patch 0007). With a GPU
device it also checks the unchanged, partial, nothing-fits, context-reduction
and preset cases with `margins`. The nothing-fits case selects one device and
reserves its total capacity so changes in free memory cannot make layers fit.
The v1.0.0 NextN case must keep the last layer back at a layer count
that `n_layers + 1` would call full. Where the device cannot run an F32
`MUL_MAT` (the paravirtual Metal device of the macOS runners has no simdgroup
reduction), llama.cpp keeps the weights on the host, and the test checks the
host's share instead. Without a GPU it checks the CPU-only case. The
`model-fit-wait` scenario of `llamadart_exit_teardown_test` checks that exit
teardown waits for a fit in flight.

`llamadart_device_memory_test` checks the Windows rules in plain functions on
every platform: the adapter segment arithmetic (aperture segments, integrated
adapters, an empty or exhausted adapter), the minimum of the two numbers, the
`Vulkan<index>` and `CUDA<index>` names, the CUDA-to-LUID mapping with fake
driver calls (each failure has its own detail), the Vulkan and CUDA branches of
the system reading, and the margins (each device gets its own correction, a
device without a reading is untouched, a device that needs one and has none
throws). `llamadart_vulkan_device_info_test` checks that the probe keeps each
device's LUID and its validity after the driver-priority reorder. On Windows
the first test also calls `D3DKMTQueryStatistics` for every DXGI adapter, which
checks the call and the buffer layout; the runners have no GPU, so not the
numbers.

CI runs the fit test on macOS, Linux, and CPU-only Windows x64 and arm64
runners. Injected Windows devices and KMT records exercise both device listings,
strict report correction, and margin routing with both CUDA and Vulkan
registered. Hardware qualification still covers Vulkan and CUDA fits while
other processes hold GPU memory, including CUDA without `ggml-vulkan.dll`.
