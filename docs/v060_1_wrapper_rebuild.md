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
| `const char * llama_dart_last_error(void)` | `NULL` when the calling thread's last call with a barrier caught nothing; otherwise valid UTF-8 of at most 511 bytes |
| `void llama_dart_clear_last_error(void)` | none |
| `bool llama_dart_sampler_accept(struct llama_sampler *, llama_token)` | `false` after a caught exception |
| `struct llama_sampler * llama_dart_sampler_init_grammar_lazy_patterns(const struct llama_vocab *, const char * grammar_str, const char * grammar_root, const char ** trigger_patterns, size_t num_trigger_patterns, const llama_token * trigger_tokens, size_t num_trigger_tokens)` | `NULL`; also llama.cpp's value for a grammar that does not parse, so the last error tells them apart |
| `int32_t llama_dart_tokenize(const struct llama_vocab *, const char * text, int32_t text_len, llama_token * tokens, int32_t n_tokens_max, bool add_special, bool parse_special)` | `LLAMA_DART_STATUS_EXCEPTION`; `llama_tokenize` returns the same `INT32_MIN` for more than `INT32_MAX` tokens, so the last error tells them apart |
| `int32_t llama_dart_token_to_piece(const struct llama_vocab *, llama_token, char * buf, int32_t length, int32_t lstrip, bool special)` | `LLAMA_DART_STATUS_EXCEPTION`, in no other case |
| `bool llama_dart_memory_clear(llama_memory_t, bool data)` | `false`; `true` otherwise (upstream returns nothing) |
| `struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_audio(size_t n_samples, const float * data)` | `NULL` |
| `struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_buf(struct mtmd_context *, const unsigned char * buf, size_t len)` | `NULL`; also for an input that cannot be decoded, so the last error tells them apart |
| `struct mtmd_bitmap * llama_dart_mtmd_bitmap_init_from_file(struct mtmd_context *, const char * fname)` | as `_from_buf` |
| `ggml_backend_t llama_dart_ggml_backend_dev_init(ggml_backend_dev_t, const char * params)` | `NULL`; also upstream's failure value |
| `bool llama_dart_ggml_backend_dev_memory(ggml_backend_dev_t, size_t * free, size_t * total)` | `false` with both outputs zero; `true` otherwise (upstream returns nothing) |
| `bool llama_dart_ggml_backend_dev_get_props(ggml_backend_dev_t, struct ggml_backend_dev_props * props)` | `false` with `props` zeroed; `true` otherwise |
| `ggml_backend_buffer_t llama_dart_ggml_backend_alloc_ctx_tensors(struct ggml_context *, ggml_backend_t)` | `NULL`; also upstream's failure value |
| `bool llama_dart_ggml_backend_tensor_set(struct ggml_tensor *, const void * data, size_t offset, size_t size)` | `false`; `true` otherwise (upstream returns nothing) |
| `bool llama_dart_ggml_backend_tensor_get(const struct ggml_tensor *, void * data, size_t offset, size_t size)` | `false`; `true` otherwise |
| `bool llama_dart_ggml_backend_sched_alloc_graph(ggml_backend_sched_t, struct ggml_cgraph *)` | `false`; also upstream's failure value |
| `bool llama_dart_ggml_backend_sched_synchronize(ggml_backend_sched_t)` | `false`; `true` otherwise |

All of them are calls in flight for exit teardown, like the existing
wrappers. The two bitmap functions from a buffer and a file take upstream's
default options and no placeholder and return the `bitmap` of upstream's
result, which is how `llamadart` calls them.

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

Object state after a caught exception, as the header states it per object: a
sampler may be reset or freed and the context it sampled from is unchanged; a
llama or mtmd context, speculative state, a TTS task and a ggml backend,
scheduler or buffer may only be freed; the model or vocabulary of
`llama_dart_tokenize`, `llama_dart_token_to_piece` and the grammar constructor
is only read and stays usable.

Windows: the presets compile with `/EHsc`, under which MSVC and clang-cl
assume that an `extern "C"` function never throws and remove a `catch` around
a call to one, which is every llama.cpp call the wrapper makes. `libllamadart`
is therefore compiled with `/EHsc-`. llama.cpp's own libraries keep `/EHsc`,
so while an exception unwinds through one of their `extern "C"` functions the
destructors of that function's locals may be skipped: the exception still
reaches the barrier, but memory or a lock may stay held. After
`LLAMA_DART_STATUS_EXCEPTION` or any other caught exception on Windows, every
object that was passed to the call must be freed and not reused, a sampler and
a model included.

Android: the NDK links the C++ runtime statically (`ANDROID_STL=c++_static`,
now passed explicitly), so a typed catch in `libllamadart.so` of an exception
thrown in `libllama.so` works only because one library of the bundle,
`libggml-base.so`, defines the runtime (`__cxa_throw`, `__cxa_begin_catch`)
and the type information of `std::exception` and `std::runtime_error`, and the
others import them from it. `tools/validate_android_artifacts.py` now fails a
bundle in which more than one library defines them, none does, or
`libllama.so` or `libllamadart.so` does not import what it throws or catches
with. `c++_shared` was not chosen: it adds `libc++_shared.so` to the bundle,
where it clashes with another plugin's or NDK's copy in the same app.

The last error lives in thread-local storage that the record itself never
allocates. The platform may allocate a thread's block on first access:
emulated thread-local storage does on Android below API 29, which is what
these builds target, and so does the dynamic loader for a library loaded with
`dlopen`. Every barrier touches the storage before the call that may throw,
so recording `std::bad_alloc` afterwards finds it there. If that first
allocation fails, the platform aborts before the call has started; that is an
out-of-memory condition the barrier does not turn into an error.

glibc cancels a thread by a forced unwind, which the barrier's `catch (...)`
would swallow and glibc then ends the process: a thread inside libllamadart
must not be cancelled.

Not covered: `GGML_ASSERT` and `GGML_ABORT` call `abort`, and a signal is not
an exception.

### Upstream functions that llamadart calls directly

Every upstream function that `lib/src/backends/llama_cpp/` of `llamadart`
calls at `ad31d529e` (0.11.1), outside the generated bindings, with what a
caller does about exceptions. With this rebuild no function in the right-hand
column needs a direct call.

Behind a barrier:

| Upstream function | Wrapper |
| --- | --- |
| `llama_model_load_from_file`, `llama_init_from_model`, `llama_decode`, `llama_encode`, `llama_synchronize`, `llama_sampler_sample`, `llama_state_save_file`, `llama_state_load_file`, `llama_state_seq_get_size_ext`, `llama_state_seq_get_data_ext`, `llama_state_seq_set_data_ext`, `llama_adapter_lora_init`, `mtmd_init_from_file`, `mtmd_tokenize`, `mtmd_encode_chunk`, `mtmd_helper_eval_chunks`, `mtmd_helper_eval_chunk_single`, `mtmd_helper_decode_image_chunk`, `ggml_backend_sched_graph_compute` | the `llama_dart_` function of the same name, since `v0.5.0-2` |
| `llama_free`, `llama_model_free`, `mtmd_free` | `llama_dart_exit_free` |
| `llama_sampler_accept`: a grammar throws for a token it rejects | `llama_dart_sampler_accept` |
| `llama_sampler_init_grammar_lazy_patterns`: `std::regex_error` for a trigger pattern that is not a valid regular expression | `llama_dart_sampler_init_grammar_lazy_patterns` |
| `llama_tokenize`: `std::out_of_range` for a byte that the vocabulary has no token for, `std::runtime_error` from the pre-tokenizer's regular expressions | `llama_dart_tokenize` |
| `llama_token_to_piece`: `std::out_of_range` for a token that is not in the vocabulary, such as `LLAMA_TOKEN_NULL` | `llama_dart_token_to_piece` |
| `llama_memory_clear`: clears the backend's buffers | `llama_dart_memory_clear` |
| `mtmd_bitmap_init_from_audio`, `mtmd_helper_bitmap_init_from_buf`, `mtmd_helper_bitmap_init_from_file`: allocate what they decode from caller-supplied media | `llama_dart_mtmd_bitmap_init_from_audio`, `_from_buf`, `_from_file` |
| `ggml_backend_dev_init`: ggml-vulkan throws `Unsupported device` and Vulkan errors while it creates the device | `llama_dart_ggml_backend_dev_init` |
| `ggml_backend_dev_memory`, `ggml_backend_dev_get_props`: ggml-vulkan lists the physical devices on each call (`enumeratePhysicalDevices`), which throws `vk::SystemError` when the loader fails | `llama_dart_ggml_backend_dev_memory`, `llama_dart_ggml_backend_dev_get_props` |
| `ggml_backend_alloc_ctx_tensors`, `ggml_backend_tensor_set`, `ggml_backend_tensor_get`, `ggml_backend_sched_alloc_graph`, `ggml_backend_sched_synchronize`: allocate or transfer on the backend, where ggml-vulkan throws `vk::SystemError` | the `llama_dart_ggml_backend_` function of the same name |

Called directly, and why that is safe. Any of them can still throw
`std::bad_alloc` where it allocates a small object.

| Upstream functions | Why no exception reaches the caller |
| --- | --- |
| `ggml_init`, `ggml_free`, `ggml_new_tensor_1d`, `ggml_new_tensor_2d`, `ggml_add`, `ggml_mul`, `ggml_mul_mat`, `ggml_norm`, `ggml_relu`, `ggml_gelu_erf`, `ggml_soft_max_ext`, `ggml_get_rows`, `ggml_cont`, `ggml_cont_2d`, `ggml_permute`, `ggml_transpose`, `ggml_reshape_3d`, `ggml_set_input`, `ggml_set_output`, `ggml_new_graph_custom`, `ggml_build_forward_expand`, `ggml_graph_overhead_custom`, `ggml_tensor_overhead` | `ggml.c` is C; a failed assertion aborts |
| `ggml_backend_load`, `ggml_backend_load_all`, `ggml_backend_load_all_from_path`, `ggml_backend_register`, and the `ggml_backend_init` and `ggml_backend_score` entry points of a backend module | `ggml-backend-reg.cpp` uses the `std::error_code` file system calls, and a module's registration catches its own exceptions (`ggml_backend_vk_reg`) |
| `ggml_backend_reg_count`, `ggml_backend_reg_get`, `ggml_backend_reg_by_name`, `ggml_backend_reg_name`, `ggml_backend_reg_dev_count`, `ggml_backend_reg_dev_get`, `ggml_backend_reg_get_proc_address`, `ggml_backend_dev_count`, `ggml_backend_dev_get`, `ggml_backend_dev_by_type`, `ggml_backend_dev_name`, `ggml_backend_dev_type`, `ggml_backend_dev_backend_reg`, `ggml_backend_get_device`, `ggml_backend_name`, and the CPU backend's thread count setter | read the registry or what a backend stored when it registered its devices; ggml-vulkan answers the name and the type from its device context |
| `ggml_backend_sched_new`, `ggml_backend_sched_reset`, `ggml_backend_buffer_set_usage` | set up or reset the scheduler's own tables; no backend call |
| `ggml_backend_free`, `ggml_backend_sched_free`, `ggml_backend_buffer_free`, `llama_sampler_free`, `llama_batch_free`, `llama_adapter_lora_free`, `mtmd_bitmap_free`, `mtmd_input_chunks_free`, `llama_backend_free` | destructors, which do not throw |
| `llama_backend_init`, `llama_context_default_params`, `llama_model_default_params`, `llama_sampler_chain_default_params`, `llama_supports_gpu_offload`, `llama_max_parallel_sequences`, `llama_batch_init`, `llama_log_get` | return constants or `malloc` |
| `llama_model_get_vocab`, `llama_model_n_ctx_train`, `llama_model_n_embd`, `llama_model_n_embd_out`, `llama_model_has_encoder`, `llama_model_has_decoder`, `llama_model_is_recurrent`, `llama_model_is_hybrid`, `llama_model_is_diffusion`, `llama_model_ftype`, `llama_model_chat_template`, `llama_model_meta_count`, `llama_model_meta_key_by_index`, `llama_model_meta_val_str`, `llama_model_meta_val_str_by_index`, `llama_adapter_get_alora_n_invocation_tokens`, `llama_adapter_get_alora_invocation_tokens` | read fields and the metadata map of a loaded model |
| `llama_n_ctx`, `llama_n_ctx_seq`, `llama_n_batch`, `llama_n_ubatch`, `llama_n_seq_max`, `llama_n_rs_seq`, `llama_n_threads_batch`, `llama_pooling_type`, `llama_get_memory`, `llama_perf_context`, `llama_perf_context_reset`, `llama_set_embeddings`, `llama_set_adapters_lora` | read or set fields of a context; the adapter list is applied by the next decode |
| `llama_memory_seq_rm`, `llama_memory_seq_pos_max` | bookkeeping of the cache cells; no backend call |
| `llama_get_logits`, `llama_get_logits_ith`, `llama_get_embeddings`, `llama_get_embeddings_ith`, `llama_get_embeddings_seq` | first wait for the backend, which `llama_dart_decode` has already done behind the barrier; llama.cpp catches its own index errors and returns `NULL`, or aborts in a Debug build |
| `llama_vocab_n_tokens`, `llama_vocab_bos`, `llama_vocab_eos`, `llama_vocab_sep`, `llama_vocab_mask`, `llama_vocab_is_eog`, `llama_vocab_get_suppress_tokens` | read fields of the vocabulary |
| `llama_vocab_get_text`, `llama_token_get_text` | throw `std::out_of_range` only for a token outside the vocabulary; both call sites pass one read from it, the BOS token after `>= 0` and the mask token after a range check |
| `llama_sampler_chain_init`, `llama_sampler_chain_add`, `llama_sampler_init_greedy`, `llama_sampler_init_dist`, `llama_sampler_init_top_k`, `llama_sampler_init_top_p`, `llama_sampler_init_min_p`, `llama_sampler_init_temp`, `llama_sampler_init_penalties`, `llama_sampler_init_logit_bias`, `llama_perf_sampler`, `llama_perf_sampler_reset` | allocate and fill a small sampler object |
| `llama_sampler_init_grammar` | the grammar parser catches its own exceptions and the function returns `NULL` (`llama-grammar.cpp` `parse`); it compiles no regular expression |
| `mtmd_context_params_default`, `mtmd_helper_init_opt_default`, `mtmd_default_marker`, `mtmd_support_vision`, `mtmd_support_audio`, `mtmd_helper_support_video`, `mtmd_log_set`, `mtmd_helper_log_set`, `mtmd_input_chunks_init`, `mtmd_input_chunks_size`, `mtmd_input_chunks_get`, `mtmd_input_chunk_get_type`, `mtmd_get_output_embd` | return constants, read fields, or create an empty list |

`llamadart` calls no chat template function of llama.cpp and neither
`llama_detokenize` nor `llama_sampler_apply`. `llama_print_system_info` is
named only by a test of the bindings stub; it appends the registered
backends' feature lists to a static string and throws nothing but
`std::bad_alloc`.

## Sampler fixes

`llama_dart_sampler_sample_and_accept_n` reimplements llama.cpp's
`common_sampler_sample_and_accept_n` and differed from it in two ways. Both
are covered by the `model-sample-accept` scenario with a sampler that records
what it is made to accept.

- A token that the context's backend sampler chose was accepted twice.
  `llama_sampler_sample` accepts such a token itself, in every llama.cpp
  release from `v0.2.0` to `v0.6.0`, and the function accepted it again:
  measured as one accept through `llama_dart_sampler_sample` and two through
  `llama_dart_sampler_sample_and_accept_n`. The function now takes the token
  from the context and accepts it once. Only a context created with
  `llama_context_params.samplers` has a backend sampler, and `llamadart`
  creates none, so its generation was not affected; the native releases
  `v0.2.0` to `v0.6.0` have the defect for a caller that does.
- The draft tokens after an end of generation were accepted. llama.cpp
  `v0.6.0` stops when the target samples an end-of-generation token that the
  draft also predicted and more draft tokens follow, because those tokens are
  not output. The function went on: for a draft of three with the end of
  generation second it returned and accepted four tokens, now two. This path
  is the one `llamadart` uses to verify a speculative draft.

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
Creating the instance loads the system's GPU drivers and layers into the
process, as registering ggml-vulkan does; a driver that crashes there cannot
be caught, so a caller asks only when it is considering Vulkan.

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

On macOS arm64 / Apple M4 Max the Debug Metal build passed 45 default CTest
cases (37 before), 183 Python tests (175 before), all 85 required wrapper
exports (65 before) and the grammar repetition boundary. The same tests pass
in a RelWithDebInfo build with AddressSanitizer and
UndefinedBehaviorSanitizer. Against the wrapper of `main`, sampling a token
that the grammar rejects through `llama_dart_sampler_sample` ended the process
with `libc++abi: terminating due to uncaught exception of type
std::runtime_error: Unexpected empty grammar stack after accepting piece: qb
(42)` and status 134; with the barrier the same program gets
`LLAMA_TOKEN_NULL` and exits with status 0.

Mutations that the tests reject: skipping `llama_dart_exit_call_end` on the
exception path of `llama_dart_sampler_accept` or of the free guard
(`barrier-grammar`, `barrier-free`), not restoring the count of frees in
flight when a free throws (`barrier-free-idle`), calling the Vulkan 1.1
queries under a 1.0 instance, not moving a replaced driver to the end of the
device list (`llamadart_vulkan_device_info_test`), not failing the TTS task
after a caught exception, and not freeing what a creating call made when
tracking throws (`llamadart_barrier_test`).

In `validate_wrapper.yml`:

- `wrapper-contract` runs every test on Linux x64 against the pinned,
  post-v0.4.0 and v0.6.0 upstreams and reads lavapipe through the wrapper
  (`llvmpipe`, API 1.4, loader 1.3, subgroup size 8, with
  `GGML_VK_VISIBLE_DEVICES=0`).
- `windows-arm64-kleidiai` runs every test that needs no backend, with
  clang-cl.
- `msvc-mtmd-link-contract` builds `libllamadart` with MSVC `cl` for x64, the
  compiler of the Windows x64 release, runs the barrier, grammar and Vulkan
  probe tests and validates the exports. Before, it built only `mtmd`.
- `android-vulkan-shaders`, on the pinned upstream, builds `libllamadart` and
  the CPU backend with NDK 28.2 in the configuration the release takes its
  Android core libraries from, and validates the exports and the bundle's
  shared exception runtime. No lane runs code on Android.
- `macos-exit-teardown` runs the exit teardown and barrier scenarios,
  including `model-barrier` and `model-sample-accept`, in Release and under
  AddressSanitizer.

Forced exceptions per wrapper family, in `barrier-free` and `barrier-grammar`
on every platform: a grammar that rejects the accepted token, a trigger
pattern `(`, `LLAMA_TOKEN_NULL` to `llama_dart_token_to_piece`, a text with a
byte the vocabulary has no token for, a ggml device whose backend throws
`Unsupported device` while it is created or while it is queried, and a free
function that throws. The bitmap constructors, `llama_dart_memory_clear` and
the ggml tensor and scheduler functions have no exception that can be forced
without a GPU backend or an allocation failure: they are run on valid and on
undecodable input, the ggml ones on the CPU backend in the macOS `graph`
scenario.
