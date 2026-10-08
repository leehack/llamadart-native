# v0.6.0-1 wrapper rebuild

Wrapper-only changes on llama.cpp `v0.6.0`
(`d81235049384534c167caea52b85a694f6103d14`) for the native rebuild tag
`v0.6.0-1`. The exported header only gains declarations: no existing symbol,
struct or enumerator is removed or changes its signature. `llamadart` adopts
the additions after the rebuild is published.

## Exception barrier

Issue: [llamadart-native#102](https://github.com/leehack/llamadart-native/issues/102).
llama.cpp throws `std::runtime_error: Unexpected empty grammar stack after
accepting piece` from `llama_grammar_accept_token` (`src/llama-grammar.cpp`
lines 1472 and 1527 at `v0.6.0`) when a grammar sampler accepts a token that
its grammar rejects. Through `llama_dart_sampler_sample` that exception
crossed the C ABI and aborted the process.

New exports:

| Symbol | Failure semantics |
| --- | --- |
| `const char * llama_dart_last_error(void)` | `NULL` when the calling thread's last call with a barrier caught nothing |
| `void llama_dart_clear_last_error(void)` | none |
| `bool llama_dart_sampler_accept(struct llama_sampler *, llama_token)` | `false` after a caught exception |

`enum llama_dart_status { LLAMA_DART_STATUS_EXCEPTION = INT32_MIN }` is the
value an `int32_t` status or count export returns after a caught exception.

Existing exports keep their signatures and gain the barrier:

| Exports | After a caught exception |
| --- | --- |
| `llama_dart_model_load_from_file`, `llama_dart_init_from_model`, `llama_dart_mtmd_init_from_file`, `llama_dart_adapter_lora_init`, `llama_dart_sampler_init_reasoning_budget`, `llama_dart_speculative_init`, `llama_dart_mtp_init`, `llama_dart_mtp_init_with_draft_model`, `llama_dart_ngram_simple_init` | `NULL`; what the call had created is freed |
| `llama_dart_tts_init` | `NULL`, `*out_status = LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR` |
| `llama_dart_decode`, `llama_dart_encode`, `llama_dart_mtmd_tokenize`, `llama_dart_mtmd_encode_chunk`, `llama_dart_mtmd_helper_eval_chunks`, `llama_dart_mtmd_helper_eval_chunk_single`, `llama_dart_mtmd_helper_decode_image_chunk`, `llama_dart_speculative_draft`, `llama_dart_mtp_draft`, `llama_dart_ngram_draft`, `llama_dart_sampler_sample_and_accept_n` | `LLAMA_DART_STATUS_EXCEPTION` |
| `llama_dart_sampler_sample` | `LLAMA_TOKEN_NULL` |
| `llama_dart_state_save_file`, `llama_dart_state_load_file`, `llama_dart_speculative_begin`, `llama_dart_speculative_process_batch`, `llama_dart_mtp_begin`, `llama_dart_mtp_process_batch`, `llama_dart_ngram_begin`, `llama_dart_ngram_process_batch`, `llama_dart_exit_track` | `false` |
| `llama_dart_state_seq_get_size_ext`, `llama_dart_state_seq_get_data_ext`, `llama_dart_state_seq_set_data_ext` | `0` |
| `llama_dart_ggml_backend_sched_graph_compute` | `GGML_STATUS_FAILED` |
| `llama_dart_tts_start`, `llama_dart_tts_step`, `llama_dart_tts_reset` | `LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR`; the task is `FAILED` and `llama_dart_tts_last_error` has the message |
| `llama_dart_tts_get_info`, `llama_dart_tts_set_cancel_flag` | `LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR` |
| `llama_dart_synchronize`, `llama_dart_speculative_accept`, `llama_dart_mtp_accept`, `llama_dart_ngram_accept` | nothing returned; `llama_dart_last_error` is set |
| `llama_dart_exit_free`, `llama_dart_tts_free`, `llama_dart_speculative_free`, `llama_dart_mtp_free`, `llama_dart_ngram_free` | nothing returned; `llama_dart_last_error` is set, and is not cleared when nothing is caught |

`NULL`, `false`, `0`, `GGML_STATUS_FAILED` and
`LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR` are also what llama.cpp returns for a
failure of its own. A function with a barrier clears the calling thread's last
error on entry, so `llama_dart_last_error() != NULL` after the call means a
caught exception. The functions that free do not clear it: a Dart
`NativeFinalizer` may run `llama_dart_exit_free` on a thread between a failed
call and the read of its error. `LLAMA_DART_STATUS_EXCEPTION` and, from
`llama_dart_sampler_sample`, `LLAMA_TOKEN_NULL` are returned in no other case.

No barrier, and the last error is left unchanged: `llama_dart_set_log_level`,
`llama_dart_exit_untrack`, `llama_dart_exit_tracked_count`,
`llama_dart_exit_call_begin`, `llama_dart_exit_call_end`,
`llama_dart_exit_set_wait_ms`, `llama_dart_exit_teardown`,
`llama_dart_speculative_get_draft_context`, `llama_dart_speculative_need_embd`,
`llama_dart_speculative_need_embd_nextn`, `llama_dart_mtp_get_draft_context`,
`llama_dart_tts_api_version`, `llama_dart_tts_request_default`,
`llama_dart_tts_cancel`, `llama_dart_tts_eval_callback`,
`llama_dart_tts_get_output_info`, `llama_dart_tts_read_pcm` and
`llama_dart_tts_last_error`. Teardown swallows an exception from a tracked
object's free function and goes on to the next object.

Exit teardown interaction: every guard of a call in flight is released while
the exception unwinds, and the wait for the backend after a decode still runs.
`llama_dart_exit_free` restores its counters when the free function throws. A
creating function owns its draft context, speculative state and handle until
the handle is tracked, so an exception leaves no untracked context, which on
Metal would abort at exit.

Windows: the presets compile with `/EHsc`, under which MSVC and clang-cl
assume that an `extern "C"` function never throws and remove a `catch` around
a call to one, which is every llama.cpp call the wrapper makes. `libllamadart`
is therefore compiled with `/EHsc-`. llama.cpp itself keeps `/EHsc`, so a
frame of one of its `extern "C"` functions between the throw and the barrier
may skip the destructors of its locals; the exception still reaches the
barrier.

Not covered: `GGML_ASSERT` and `GGML_ABORT` call `abort`, and a signal is not
an exception. Upstream functions that a caller reaches without a `llama_dart_`
wrapper have no barrier. `llamadart` calls these directly today:
`llama_sampler_accept` (use `llama_dart_sampler_accept`),
`llama_sampler_init_grammar_lazy_patterns` (a malformed trigger pattern throws
`std::regex_error`), `llama_tokenize`, `llama_token_to_piece` (an out-of-range
token throws `std::out_of_range`, which includes `LLAMA_TOKEN_NULL`) and the
`mtmd` bitmap constructors.

## Vulkan device facts

`llamadart` needs two facts about the device ggml-vulkan will use before it
uses it: the subgroup size, because the small matmul tile is wrong at subgroup
size 16 ([llamadart#948](https://github.com/leehack/llamadart/issues/948)),
and the device's Vulkan API version, because ggml-vulkan calls a null
`vkGetBufferDeviceAddress` on a driver below 1.2
([llamadart#782](https://github.com/leehack/llamadart/issues/782)).

New exports:

| Symbol | Failure semantics |
| --- | --- |
| `int32_t llama_dart_vulkan_get_device_count(void)` | number of devices, or a negative `llama_dart_vulkan_status` |
| `int32_t llama_dart_vulkan_get_device_info(int32_t index, struct llama_dart_vulkan_device_info * out_info)` | `LLAMA_DART_VULKAN_STATUS_OK`, or a negative status; `out_info` is written only for `OK` |

`enum llama_dart_vulkan_status`: `OK` 0, `INVALID_ARGUMENT` -1 (null
`out_info` or a `struct_size` that is too small), `UNSUPPORTED` -2 (Apple
platforms, where there is no ggml-vulkan), `NO_LOADER` -3, `LOADER_ERROR` -4
(no instance, as without any driver), `NO_DEVICE` -5 (index out of range).

`struct llama_dart_vulkan_device_info`: `struct_size`,
`instance_api_version`, `physical_device_index`, `api_version`,
`driver_version`, `vendor_id`, `device_id`, `device_type`, `subgroup_size` (all
`uint32_t`) and `char device_name[256]`.

The probe opens the loader at run time (`vulkan-1.dll`, `libvulkan.so.1`,
`libvulkan.so` on Android), so `libllamadart` gains no Vulkan link dependency
and the symbols exist in every build, whichever backends its bundle has. It
resolves every entry point through `vkGetInstanceProcAddr` and checks it before
use, calls `vkGetPhysicalDeviceProperties2` and `vkGetPhysicalDeviceFeatures2`
only when both the loader and the device are at least Vulkan 1.1, and creates
an instance but no logical device. The result is read once per process.

Device order at `d81235049` (`ggml_vk_instance_init`,
`ggml/src/ggml-vulkan/ggml-vulkan.cpp`), which the probe mirrors so that index
`N` is ggml's `VulkanN`:

1. With `GGML_VK_VISIBLE_DEVICES`, exactly the listed loader indices in the
   listed order; an index out of range leaves ggml without devices.
2. Otherwise every discrete or integrated GPU with `storageBuffer16BitAccess`,
   in loader order. Two entries with the same `deviceUUID`, or the same valid
   `deviceLUID`, are one GPU under two drivers, except when both are MoltenVK:
   the preferred driver stays, and moves to the end of the list when it
   replaces the other.
3. Otherwise the first device that is not a CPU.

ggml refuses to initialize when `vkEnumerateInstanceVersion` is below 1.2 and
then registers no device; the probe still lists the devices, with
`instance_api_version`. ggml sets `device->subgroup_size` from
`VkPhysicalDeviceSubgroupProperties.subgroupSize` and the small matmul tile
compares that value, which is what `subgroup_size` reports; the minimum and
maximum of `VK_EXT_subgroup_size_control` only choose the required subgroup
size of individual pipelines.

Known differences: ggml reads the 16-bit storage feature from
`VkPhysicalDeviceVulkan11Features`, which a driver below 1.2 does not fill in,
so its choice for such a device next to other GPUs is undefined; the probe
reads the equivalent `VkPhysicalDevice16BitStorageFeatures`. ggml holds 16
devices, and so does the probe. A single GPU is index 0 either way.

Not verified on hardware: a Mali device with subgroup size 16, a device or
loader below Vulkan 1.2, any Android or Windows loader, and the agreement of
the mirrored order with a real ggml-vulkan on a machine with several GPUs.
The table-driven test covers those cases against the code that
`ggml_vk_instance_init` was read to do, not against ggml-vulkan itself.

## Exit teardown on Linux: not in this rebuild

Issue: [llamadart#949](https://github.com/leehack/llamadart/issues/949). On
Linux a C `exit()` while another thread is inside llama.cpp crashes that
thread: `exit()` destroys function-local C++ statics that it still reads
(`ggml_backend_cpu_get_extra_buffer_types` in `ggml-cpu/ggml-cpu.cpp`, the
map behind `unicode_utf8_to_byte` in `src/unicode.cpp`). The exit registry
is armed on Apple platforms only. Arming it on Linux was left out because it
does not fix the crash as it stands, and what would fix it is a change to
every library of the bundle that no lane here can prove.

Why the Apple mechanism does not carry over:

- `exit()` runs one process-wide list of `atexit` and `__cxa_atexit` entries
  in reverse order of registration, on glibc, musl and bionic alike, and the
  C++ standard requires that interleaving. The destructor of a function-local
  static is registered when the static is first constructed, whichever library
  it belongs to. Library destructors (`.fini_array`,
  `__attribute__((destructor))`) run from `_dl_fini`, the first entry, and so
  last.
- A handler that libllamadart registers therefore runs before the statics
  that existed when it was registered and after every static constructed
  later. The map behind `unicode_utf8_to_byte` is first constructed inside a
  model load (`llama-vocab.cpp:3439`), so for an exit during the first load a
  handler registered before the load runs after the map is destroyed, and one
  registered after the load is not registered yet. That is the `quit-loading`
  frame of the issue.
- On Apple the wrapper solves this by defining `__cxa_atexit` inside its own
  image, so that the destructor of every static in the image runs teardown
  first. That works because llama.cpp and ggml are linked into
  `libllamadart.dylib`. On Linux, Android and Windows they are separate
  libraries (`libllama.so`, `libggml-base.so`, `libggml-cpu.so` and the other
  backend modules), whose registrations bind to the C library's
  `__cxa_atexit`. A library loaded with `dlopen` cannot interpose that.
- Registering the plain `atexit` handler when the first object is tracked, as
  the Apple build also does, would cover only the statics constructed before
  that moment: likely the `quit-generating` frame, since the CPU buffer type
  list is built during the load, but not `quit-loading`, and not a static that
  the first decode, sample or detokenization constructs. It would also free
  Vulkan and CUDA objects during `exit()` in an order relative to the driver's
  own exit handlers that nothing controls, where exiting with an idle model
  is clean today (8 of 8 in the issue). No Linux host was available to
  measure either effect, and the Linux lanes here load no backend, so they run
  no model scenario.

Recommended design, for a later rebuild:

1. Compile a small object into every shared library of the bundle that holds
   llama.cpp statics (`ggml-base`, `ggml`, each `ggml-cpu` variant and backend
   module, `llama`, `mtmd`, `llama-common`), from this repository's CMake and
   without patching upstream. It defines a hidden `__cxa_atexit`, as
   `src/llama_dart_wrapper.cpp` does for Apple, that registers each destructor
   behind a call to one gate function exported by the lowest library,
   `libggml-base.so`. libllamadart installs teardown as that gate's callback.
2. On Linux the gate should only wait for the calls in flight and block new
   ones, without freeing tracked objects: nothing on Linux aborts over a live
   buffer, which is the reason Apple frees them, and freeing GPU objects late
   in `exit()` is the riskier half.
3. Prove it with a Linux lane that loads the CPU backend module and runs the
   `model-load-wait` and `model-decode-wait` scenarios of
   `tests/exit_teardown_test.cpp`, with AddressSanitizer, plus the
   `quit-loading` and `quit-generating` probes on a GPU host for Vulkan and
   CUDA.

A call that is not a call in flight is not waited for under any of these, so
a long native call that bypasses the `llama_dart_` wrappers stays exposed.
The NVIDIA driver frame of the image-generation probe belongs to
`stable-diffusion-native`. A native host can already call
`llama_dart_exit_teardown` itself before `exit`.

Windows, from the documented C runtime behavior and not from a run: `exit()`
ends in `ExitProcess`, which terminates the other threads before any DLL
receives `DLL_PROCESS_DETACH`, and each DLL destroys its statics there. A
worker is gone before the statics it read are destroyed, so this crash has no
counterpart, and a handler in `llamadart.dll` would run too late to wait for
anything.

## Checks

```bash
git submodule update --init --recursive
cmake -S . -B build/v060 -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_METAL=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/v060 --parallel 8
ctest --test-dir build/v060 --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/validate_exports.py --format nm --tool nm \
  --forbid-import __cxa_atexit build/v060/libllamadart.dylib
python3 tools/validate_grammar_boundary.py build/v060/libllamadart.dylib
```

On macOS arm64 / Apple M4 Max the Debug Metal build passed 42 default CTest
cases (37 before), 175 Python tests, all 70 required wrapper exports (65
before) and the grammar repetition boundary. Against the wrapper of `main`,
sampling a token that the grammar rejects through
`llama_dart_sampler_sample` ended the process with `libc++abi: terminating
due to uncaught exception of type std::runtime_error: Unexpected empty grammar
stack after accepting piece: qb (42)` and status 134; with the barrier the
same program gets `LLAMA_TOKEN_NULL` and exits with status 0.

The barrier scenarios also fail when a call in flight is not ended: making
`llama_dart_sampler_accept` or the free guard skip `llama_dart_exit_call_end`
on the exception path fails `barrier-grammar` and `barrier-free`.

An Android arm64 Release build of `libllamadart.so` with NDK 28.2 (CPU only)
compiled both sources and exports the 70 symbols. No pull request lane
compiles the wrapper for Android, and none compiles it with MSVC `cl` for
Windows x64; the Windows ARM64 lane uses clang-cl. Both are first built by
the release workflow.

In `validate_wrapper.yml`, `wrapper-contract` runs every test on Linux x64
against the pinned, post-v0.4.0 and v0.6.0 upstreams and reads lavapipe
through the wrapper (`llvmpipe`, API 1.4, loader 1.3, subgroup size 8, with
`GGML_VK_VISIBLE_DEVICES=0`); `windows-arm64-kleidiai` runs every test that
needs no backend; `macos-exit-teardown` runs the exit teardown and barrier
scenarios, including `model-barrier`, in Release and under AddressSanitizer.
