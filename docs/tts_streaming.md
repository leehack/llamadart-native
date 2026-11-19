# Streamed TTS PCM

`llama_dart_tts_start_streaming` starts a task whose PCM is readable while it
generates (llamadart-native
[#198](https://github.com/leehack/llamadart-native/issues/198)). It ships in the
wrapper rebuild `v1.0.0-1`, over llama.cpp `b1b770d9`.

## Contract

```c
enum llama_dart_tts_status llama_dart_tts_start_streaming(
    struct llama_dart_tts * tts,
    const struct llama_dart_tts_request * request,
    const char * voice);   // NULL is the model's default voice
```

- `LLAMA_DART_TTS_CAPABILITY_STREAMING_PCM = 1u << 3` is set for Qwen3-TTS,
  LFM3-Audio and Pocket TTS, and clear for Soprano and Chatterbox.
- It validates like `llama_dart_tts_start_with_voice`. Without the bit it
  returns `UNSUPPORTED`, sets `llama_dart_tts_last_error` and leaves the task
  and the llama context untouched, so the same task can run a plain start.
- For a streamed task, `llama_dart_tts_get_output_info` and
  `llama_dart_tts_read_pcm` are also valid in `PROCESSING_PROMPT` and
  `GENERATING`. `sample_count` is the samples produced so far, samples below it
  never change, and `read_pcm` keeps its offset semantics: read from the offset
  you last reached.
- Each step appends what the decoder produced since the last one, which is
  nothing until its window fills. The step that ends generation (end of
  speech or `max_frames`) flushes the decoder and appends its tail. The
  `COMPLETED` PCM equals that of `llama_dart_tts_start` with the same request
  and seed.
- After `CANCELLED`, `FAILED` and `llama_dart_tts_reset`, both reads return
  `INVALID_STATE`.
- A task started with `start` or `start_with_voice` is unchanged: reads before
  `COMPLETED` return `INVALID_STATE`.
- Both reads run inside `llama_dart_exit_call` for every task, so exit teardown
  waits for them. This also guards a read after `COMPLETED`, which was not
  guarded before. Do not overlap a read with `llama_dart_tts_step` on the same
  task, as for every task call.

## ABI delta for the Dart bindings

Additive. No struct layout, status value or `LLAMA_DART_TTS_API_VERSION`
(still 1) changes, so an older core keeps working on a newer framework and the
reverse needs only the symbol probe.

| Change | Detail |
| --- | --- |
| `llama_dart_tts_start_streaming` | New export, resolve it lazily. Its absence means no streaming. |
| `LLAMA_DART_TTS_CAPABILITY_STREAMING_PCM` | Bit 8 of `llama_dart_tts_info.capabilities`; Qwen3-TTS, LFM3-Audio and Pocket TTS only. |
| `llama_dart_tts_get_output_info`, `llama_dart_tts_read_pcm` | Valid while a streamed task generates, see above. |

## Cadence

Upstream `stream` mode (`mtmd_helper_gen_audio_inp.stream`) changes only when
`get_output` returns audio; the decode calls happen at the same points in both
modes, so the concatenation equals the plain PCM by construction. Measured with
the llama.cpp `b1b770d9` helpers, seed 1234, macOS arm64 CPU.

| Family | First samples | Chunk | Streams |
| --- | --- | --- | --- |
| Qwen3-TTS | frame 72 | 138,240 samples (5.76 s), tail at flush | yes |
| Pocket TTS | frame 8, and at each text-chunk end | 15,360 samples (0.64 s) | yes |
| LFM3-Audio | frame 1 | 1,440 first, then 1,920 samples (0.08 s), tail at flush | yes |
| Soprano | after flush | whole utterance | `UNSUPPORTED` |
| Chatterbox | after flush | whole utterance | `UNSUPPORTED` |

An utterance shorter than a Qwen3-TTS window arrives as one chunk at the end.

## Not exposed

- Upstream Drama 3 streams 7-frame batches and decodes up to 64 otherwise, so
  its equality is not by construction. The wrapper does not drive it.
- WAV output (`MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV`): the wrapper always asks for
  float PCM.

## Checks

`llamadart_tts_stream_test` covers the readable-state and delta-append helpers
without a model. The local smokes (`llamadart_tts_smoke`,
`llamadart_tts_lfm3_smoke`, `llamadart_tts_voice_smoke`,
`llamadart_soprano_tts_smoke`) stream with real weights: window sizes, growth,
unchanged earlier samples, equality with the plain PCM, withdrawal on cancel,
and the `UNSUPPORTED` refusal.
