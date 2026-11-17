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
- `mparams` and `cparams` are written only on `SUCCESS`.
- The report is measured at the fitted parameters. It holds at most
  `LLAMA_DART_FIT_MAX_DEVICES` devices.

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
It applies to every `LLAMADART_UPSTREAMS` row; upstream's scope is the same
there.

What remains: `llama_log_set` writes the callback and its user data
separately. A thread that logs in the instant a fit installs or restores the
logger can call the fit's callback with the previous user data (the fit's
callback ignores it) or the original callback with new user data (the same
value, as fits are serialized). A thread that calls `llama_log_set` itself
during a fit has its logger overwritten when the fit restores the original.
The patch does not remove the data race in `llama_log_set`, and TSan reports it.

Teardown cannot cancel a fit, because its `no_alloc` loads report no progress.
It waits for a running fit, up to the exit wait limit.

## Upstream compatibility

CMake probes `LLAMADART_FIT_HAS_PATH_LIST` (v1.0.0 passes a list of paths) and
`LLAMADART_FIT_HAS_NGL_SKIP` (v1.0.0 reports skipped NextN layers), so the same
source builds on every `LLAMADART_UPSTREAMS` row.

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
change `n_ctx`) everywhere, and fits that overlap each other while a sentinel
logger is installed: the logger must be the same one afterwards. With a GPU
device it also checks the unchanged, partial, nothing-fits, context-reduction
and preset cases against a budget set through `margins`, and on v1.0.0 a model
with a NextN layer whose fit must keep the last layer back at a layer count
that `n_layers + 1` would call full. Where the device cannot run an F32
`MUL_MAT` (the paravirtual Metal device of the macOS runners has no simdgroup
reduction), llama.cpp keeps the weights on the host, and the test checks the
host's share instead. Without a GPU it checks the CPU-only case. The
`model-fit-wait` scenario of `llamadart_exit_teardown_test` checks that exit
teardown waits for a fit in flight.

CI runs the test on the macOS runners (Metal), on Linux and on Windows arm64
(CPU-only). No CUDA, Vulkan, HIP or Windows x64 GPU run executes the fit; that
needs the hardware runs of the llamadart plan.
