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
