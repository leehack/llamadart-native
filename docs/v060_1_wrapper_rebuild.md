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
| `llama_dart_synchronize`, `llama_dart_exit_free`, `llama_dart_tts_free`, `llama_dart_speculative_free`, `llama_dart_mtp_free`, `llama_dart_ngram_free`, `llama_dart_speculative_accept`, `llama_dart_mtp_accept`, `llama_dart_ngram_accept` | nothing returned; `llama_dart_last_error` is set |

`NULL`, `false`, `0`, `GGML_STATUS_FAILED` and
`LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR` are also what llama.cpp returns for a
failure of its own. A function with a barrier clears the calling thread's last
error on entry, so `llama_dart_last_error() != NULL` after the call means a
caught exception. `LLAMA_DART_STATUS_EXCEPTION` and, from
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
