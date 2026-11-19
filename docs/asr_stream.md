# Live ASR session

`llama_dart_asr_stream_*` turns 16 kHz mono PCM into text while it arrives, for
[llamadart-native#148](https://github.com/leehack/llamadart-native/issues/148)
("Live ASR") and the Dart consumer
[llamadart#327](https://github.com/leehack/llamadart/issues/327). The header has
the full contract; this page has what a binding needs to know, why the wrapper
owns the loop, and the local check.

The session wraps the experimental `mtmd_audio_stream_*` API of llama.cpp.
Native CI qualifies the pinned upstream shipped by the current llamadart core
minor. The wrapper adds exports without changing existing exports, structs
or versions. `LLAMA_DART_ASR_STREAM_API_VERSION` is 1.

## Why a session

Calling `mtmd_audio_stream_*` from Dart is ruled out by the exit-teardown
contract (`llama_dart_wrapper.h`, "What teardown waits for"):

- A live stream owns backend memory: the encoder state is allocated on the
  projector's backend, a Metal buffer when the projector runs on the GPU. Left
  alive at C `exit` it is the `GGML_ASSERT([rsets->data count] == 0)` abort. The
  session is tracked in `LLAMA_DART_EXIT_STAGE_SESSION` before `init` returns,
  so teardown frees it before the mtmd context and the model.
- `mtmd_audio_stream_step` is not a call in flight and runs the audio encoder.
- One upstream step is unbounded: it encodes the whole backlog. 297 s pushed at
  once gave a 2.7 s step and +2.4 GB of peak memory, against the 2 s that
  teardown waits. Fed 320 ms at a time the longest step was 16 ms.
- Encode and decode can throw, and only `llama_dart_` functions have the
  exception barrier.
- Feeding the sampled token back, restarting the sequence and the logit bias are
  model details that upstream marks "EXPERIMENTAL, subjected to breaking
  changes".

So the session owns the decode and the greedy sampling, and a `step` is one
bounded unit of work: at most one step of audio from the queue to the model
(320 ms, or one 960 ms chunk), one upstream step, and when that returns rows,
one `llama_decode` and one sample. It is a single call in flight, which bounds
teardown and cancel latency. On `TOKEN_DELAY` (Voxtral Realtime), the first
producing step decodes the 39-row prefix in one `llama_decode` (~0.69 s on an
M4 Max CPU); on a ~3x slower CPU that single prefill can approach or exceed the
2 s teardown wait (splitting that prefix decode is deferred).

## Modes

The model decides the mode, by the two tests upstream's CLI and server route on.
A caller lists the modes it handles in `accepted_modes`.

| Mode | Test | Model | Text |
| --- | --- | --- | --- |
| `TOKEN_DELAY` | stream, no chat | Voxtral Realtime (`voxtral_rt`) | One token per 80 ms row, 480 ms behind the audio. The context grows 12.5 positions per second and never shifts. |
| `PROMPTED_CHUNKS` | stream and an `asr` chat template | LFM3-Audio (`lfm3a`) | The text of each 0.96 s chunk. The session passes the tokens of the `asr` template as the stream prompt and installs `mtmd_helper_audio_stream_get_logit_bias`. |

Refused with `UNSUPPORTED`: a projector without a stream (Qwen3-ASR and every
other model), a model that only prefills a chat turn (Gemma 4, LFM3-Omni,
LFM3-Audio without its `asr` template), a mode outside `accepted_modes`, audio
that is not 16 kHz, an M-RoPE decoder, and an `n_batch` below the first batch
(39 rows for Voxtral Realtime). `llama_dart_asr_stream_get_info` reads this
without creating a session, and `init` is authoritative. Gemma 4 has not been
run: it is told apart by having no `asr` template, and a model that had one
would fail in `init` with `UPSTREAM_ERROR`.

## What a binding must know

- **Input.** `push` copies into a queue bounded by `params.max_buffered_ms`
  (30 s by default) and reports how many samples it took. It never drops audio:
  a binding offers the rest again after steps drained the queue, which is the
  backpressure. Non-finite samples are refused. Only 16 kHz mono float is
  accepted; resampling stays with the caller.
- **Loop.** `step` until the state is `FINISHED`: push when the state is
  `NEEDS_AUDIO`, `finish` after the last chunk. `read_text` after every step.
- **Text.** UTF-8, complete code points only, control tokens skipped, no
  terminating zero. A token that ends inside a character waits for the next one;
  at a segment boundary, an incomplete trailing UTF-8 prefix is discarded while
  preserving all complete unread bytes from earlier segments; at stream end, an
  incomplete tail or invalid byte is read as U+FFFD.
- **Segments.** `PROMPTED_CHUNKS` restarts its sequence at a quiet chunk after
  15 s and always after 30 s of audio rows, and a word at a hard cut can be lost
  (upstream's behavior). `progress.segment_breaks` counts the restarts; the text
  read after the step that raised it belongs to the new segment, so a binding
  that reads after every step knows where each segment starts. Upstream's server
  writes a newline there, the wrapper leaves the choice to the caller.
- **Timing.** `info.delay_ms` is the model-derived delay for `TOKEN_DELAY` and
  the nominal 960 ms chunk cadence for `PROMPTED_CHUNKS`. First visible text in
  `PROMPTED_CHUNKS` needs more audio (~1.1–1.3 s measured on LFM3-Audio) because
  mel feature extraction requires a complete lookahead window.
- **Sequence ownership.** Successful `init` and `reset` claim and clear
  `params.sequence_id` immediately, and segment restarts and session release
  clear it too; the caller must dedicate that sequence to the session.
- **End.** `FINISHED` follows `finish` once the text is out, or an
  end-of-generation token of the model, which drops the audio still queued
  (`samples_buffered` shows it). `CONTEXT_FULL` (state `FINISHED`) is the typed
  failure for a context with no room for the next rows: the text so far stays
  readable. `info.max_session_ms` is the audio a `TOKEN_DELAY` session can take,
  (context size - 39 prefix rows) x 80 ms: 5.4 minutes at 4096, 10.9 at 8192.
  For `PROMPTED_CHUNKS`, `info.max_session_ms` is 0 ("no fixed duration
  estimate"): text and closing tokens (up to 48 text tokens per ordinary chunk,
  96 for a final chunk) also consume context positions alongside audio rows, so
  a small context can still finish with `CONTEXT_FULL` before a 30 s restart.
  Upstream's server does the same, because context shift is disabled with a
  projector.
- **Cancel.** `cancel` from any thread, or a caller-owned byte through
  `set_cancel_flag`, lands at the next step entry: at most one step, 0.1 s for
  LFM3-Audio and 0.5 s for Voxtral Realtime on a CPU. The session then releases
  the stream and the sequence and produces no more text. `reset` works in every
  state, clears any prior cancel at entry before releasing and reopening the
  stream (so a `cancel` after that boundary survives until the next step), and
  makes the session ready for a new input.
- **Tracking.** Every call but `cancel`, `last_error` and the three constant
  getters runs inside the guard, so teardown waits for it. `free` or
  `llama_dart_exit_free` releases a session, and `llama_dart_exit_tracked_count`
  returns to its value before `init`.
- **Not exposed.** `mtmd_audio_stream_flush(end = false)`: it pads silence and
  changed the LFM3 transcript in the probe, a comma lost, so a pause is not
  transcript-neutral.
- **Other calls on the llama context** stay idle until the session ends, is
  cancelled or is reset.

## Tests

| Test | Needs | Covers |
| --- | --- | --- |
| `llamadart_asr_stream_api_test` | nothing | Version, mode mask, defaults, null handles, wrong `struct_size`, no-op `free` and `cancel`. |
| `llamadart_asr_stream_meter_test` | nothing | Mode classification, size formulas, the queue bound and the one-step chunk, UTF-8 boundaries. |
| `llamadart_exit_teardown_test model-asr-stream`, `-untracked`, `-refused` | models, see below | The session is tracked and freed before the mtmd context, untracked contexts leave it alone, a projector without a stream is refused and tracks nothing. |
| `llamadart_asr_stream_smoke` | models and audio | Transcripts, chunking, bounds, cancel, reuse, restarts, context exhaustion, leaks. |

The exit-teardown scenarios are registered by CMake when the cache entries
`LLAMADART_ASR_STREAM_TEST_MODEL` and `_MMPROJ` (a streaming model) and
`LLAMADART_ASR_STREAM_REFUSED_MODEL` and `_MMPROJ` (Qwen3-ASR) are set. Like
`model-tts` they need real files and are not part of CI.

## Local smoke

```bash
cmake -S . -B build/asr-stream-smoke -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMADART_BUILD_ASR_STREAM_SMOKE=ON
cmake --build build/asr-stream-smoke --target llamadart_asr_stream_smoke
build/asr-stream-smoke/llamadart_asr_stream_smoke \
  /path/to/LFM3-Audio-2B-Q4_K_M.gguf \
  /path/to/mmproj-LFM3-Audio-2B-Q8_0.gguf \
  /path/to/jfk.wav [--gpu] [--threads 4] [--max-step-ms 500]
```

`jfk.wav` is `packages/llamadart_validation/assets/speech/jfk.wav` of
[llamadart](https://github.com/leehack/llamadart), 11 s. It checks, in order:

- the transcript of 320 ms chunks, compared without punctuation (CPU and Metal
  differ by a comma);
- the same text for five random chunkings (1 ms to 1.5 s), each after a `reset`;
- the whole long input pushed in one call (297 s, or the context for
  `TOKEN_DELAY`): the longest step under `--max-step-ms`, and peak memory within
  100 MB of the 11 s run;
- 44 s of input: one segment break for `PROMPTED_CHUNKS`, none for
  `TOKEN_DELAY`, and the sentence intact in at least three of the four
  repetitions;
- a cancel through the flag after four tokens, no more text afterwards, no more
  audio accepted, and the same transcript after `reset`;
- a cancel before the first step;
- `TOKEN_DELAY` in a 512-token context: `CONTEXT_FULL`, state `FINISHED`, text
  kept;
- ten open, step and free cycles with `llama_dart_exit_tracked_count` back where
  it started.

Results on an M4 Max, CPU with 4 threads, generic `GGML_NATIVE=OFF` build,
files as in [lfm3_support.md](lfm3_support.md) and the plan's Voxtral
[ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF](https://huggingface.co/ggml-org/Voxtral-Mini-4B-Realtime-2602-GGUF)
at `0b0f7ac9` (`Voxtral-Mini-4B-Realtime-2602-Q4_K_M.gguf`,
`mmproj-Voxtral-Mini-4B-Realtime-2602-Q8_0.gguf`):

| | LFM3-Audio-2B Q4_K_M (`PROMPTED_CHUNKS`) | Voxtral-Mini-4B-Realtime Q4_K_M (`TOKEN_DELAY`) |
| --- | --- | --- |
| 11 s clip, real-time factor (warm) | 0.09 | 0.61 (the first run pages the model in) |
| Longest step, 11 s clip | 133 ms | 686 ms |
| 297 s in one `push`: real-time factor, longest step | 0.09, 154 ms | 0.68, 519 ms (the context holds 324 s) |
| Peak memory, 11 s run to 297 s run | 2560 to 2626 MB | 6609 to 6609 MB |
| Segment breaks, 44 s of input | 1 (sentence intact 3 of 4 times) | 0 (4 of 4) |
| `CONTEXT_FULL` in a 512-token context | after small context (~64 tokens), text kept | after 474 tokens, text kept |

Without the metering, one upstream step for 297 s of LFM3-Audio took 2.7 to 3 s
and peak memory reached 5 GB (the smoke fails its bound at 2960 ms when the
queue hands upstream everything). The LFM3-Audio transcript of the generic CPU
build lacks the comma after "for you" that the arm64-native probe build has;
the smoke compares without punctuation. The Voxtral bound is `--max-step-ms
1000`: its decode of one step on a CPU is about 0.5 s.
