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
- With no GPU device the call succeeds, changes nothing and reports
  `n_devices == 0`.
- On success `mparams` points into the caller's `tensor_split` and
  `tensor_buft_overrides` buffers. Keep them alive until the model is loaded.
  The override patterns are static storage inside `llama-common`: never free
  them.
- `mparams` and `cparams` are written only on `SUCCESS`.
- The report is measured at the fitted parameters. It holds at most
  `LLAMA_DART_FIT_MAX_DEVICES` devices.

## Concurrency

`common_fit_params` swaps the global llama log callback through a stack object
while it measures and is documented as not thread safe. The wrapper serializes
fits and `llama_dart_set_log_level` on one mutex. A thread that calls
`llama_log_set` directly, or that logs through the callback at the instant the
fit restores it, is outside that guarantee; llamadart runs engines in several
isolates and only reaches the callback through the wrapper.

## Upstream compatibility

CMake probes `LLAMADART_FIT_HAS_PATH_LIST` (v1.0.0 passes a list of paths) and
`LLAMADART_FIT_HAS_NGL_SKIP` (v1.0.0 reports skipped NextN layers), so the same
source builds on every `LLAMADART_UPSTREAMS` row.

## ABI delta for the Dart bindings

New exports `llama_dart_fit_api_version` and `llama_dart_fit_params`, new types
`llama_dart_fit_status`, `llama_dart_fit_device` and `llama_dart_fit_report`
(`LLAMA_DART_FIT_MAX_DEVICES` is 16). Nothing existing changes. The required
export list in `tools/validate_exports.py` grows from 94 to 96.

## Tests

`llamadart_fit_api_test` writes its own 36 MiB model. It checks the contract
(null arguments, missing and unreadable models) everywhere. With a GPU device it
also checks the unchanged, partial, nothing-fits, context-reduction and preset
cases against a budget set through `margins`; without one it checks the
CPU-only case. Everywhere it runs fits beside `llama_dart_set_log_level` on
several threads. The `model-fit-wait` scenario of `llamadart_exit_teardown_test`
checks that exit teardown waits for a fit in flight. NVIDIA, Vulkan and Windows
behavior is not covered here; it needs the hardware runs of the llamadart plan.
