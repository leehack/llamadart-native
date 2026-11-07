# Chatterbox, Pocket TTS and predefined voices

Upstream stable v0.9.0 (`cd959bae`) adds Chatterbox Turbo and Multilingual V3
([#30993](https://github.com/ggml-org/llama.cpp/pull/30993), `b97627967`) and
predefined voices for Pocket TTS
([#31429](https://github.com/ggml-org/llama.cpp/pull/31429), `d02b4c0d6`).
The wrapper drives both generators opt-in and lists and selects a projector's
predefined voices. Owner issues:
[#164](https://github.com/leehack/llamadart-native/issues/164) (Chatterbox) and
[#163](https://github.com/leehack/llamadart-native/issues/163) (Pocket TTS).

## ABI delta for the Dart bindings

Additive; no struct layout, status value or `LLAMA_DART_TTS_API_VERSION`
changes.

| Change | Detail |
| --- | --- |
| `LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX = 4` | Reported only to `_ext` callers whose mask has `1u << 4`. |
| `LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS = 5` | Reported only to `_ext` callers whose mask has `1u << 5`. |
| `llama_dart_tts_supported_model_types()` | `0x3e` on v0.9.0 (was `0xe`). |
| `llama_dart_tts_get_voice_count` | New export, see below. |
| `llama_dart_tts_get_voice_name` | New export, see below. |
| `llama_dart_tts_start_with_voice` | Validates against the listed voices for every type. |

```c
enum llama_dart_tts_status llama_dart_tts_get_voice_count(
    const struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    int32_t * out_count);

enum llama_dart_tts_status llama_dart_tts_get_voice_name(
    const struct mtmd_context * mtmd,
    uint32_t accepted_model_types,
    int32_t index,
    char * out_name,       // NULL with out_capacity 0 queries the length
    size_t out_capacity,   // must exceed the length (room for the NUL)
    size_t * out_length);  // length without the NUL
```

Both take the same mask as `llama_dart_tts_get_info_ext`, so a caller sees no
voices of a generator it did not opt into (`UNSUPPORTED`, count 0). A model
without `LLAMA_DART_TTS_CAPABILITY_VOICE` has count 0. Index 0 is the voice a
request without one speaks with. Both run behind the exception barrier: a
caught exception returns `UPSTREAM_ERROR`. On a wrapper older than this change
the symbols are absent; resolve them lazily, as for the `_ext` exports.

## Rules per model type

| | Chatterbox Turbo | Chatterbox Multilingual | Pocket TTS | LFM3-Audio |
| --- | --- | --- | --- | --- |
| Capabilities | `VOICE`, `SPEAKER_REFERENCE` | `LANGUAGE`, `VOICE`, `SPEAKER_REFERENCE` | `VOICE` if the projector has voices, `SPEAKER_REFERENCE` if it has an encoder | `LANGUAGE`, `VOICE` |
| Voices | `default` (from `conds.pt`) | `default` | 25 in the Kyutai packs | `female`, `male` |
| No voice, no reference | first voice | first voice | first voice; unconditioned if none | template default |
| Voice + reference | `INVALID_ARGUMENT` | `INVALID_ARGUMENT` | `INVALID_ARGUMENT` | `UNSUPPORTED` (reference) |
| Language | `UNSUPPORTED` | prefixed as `[xx]`, not validated | `UNSUPPORTED` | chat template |
| Sampling | request settings | request settings | the weights' own | the GGUF's |
| Sequences | one | task sequence and `n_seq_max / 2 + sequence_id` | one | one |

A voice is a voice bitmap from `mtmd_gen_audio_get_voice` passed as the speaker
reference, as `llama-tts` and `llama-server` do; LFM3-Audio keeps naming its
voice in the `tts` chat template. A name matches in any case, as upstream's
lookup does, and an unknown name is `INVALID_ARGUMENT`: the wrapper never falls
back to the default. Upstream Chatterbox fails without a voice or a reference,
so a Chatterbox projector without voices needs a reference
(`INVALID_ARGUMENT` otherwise). Pocket TTS without either speaks unconditioned,
as upstream allows.

Chatterbox Multilingual runs classifier-free guidance on a second sequence.
The task owns it like its own sequence: it is cleared at start, on completion,
error, cancel, reset and free. `llama_dart_tts_init_ext` refuses a context
with `n_seq_max < 2`, and a `sequence_id` not below `n_seq_max / 2` is
`INVALID_ARGUMENT`, before the context is touched. A unified KV cache is not
needed; without one, each sequence holds `n_ctx / n_seq_max` positions. The
wrapper tells Multilingual from Turbo as the bundled helper does at v0.9.0, by
the `a.gen.t3.text_pos_emb` table. Upstream
[#31698](https://github.com/ggml-org/llama.cpp/pull/31698) (after v0.9.0)
moves that to `{arch}.classifier_free_guidance`; the bump that takes it must
move this check with it.

## Sampling and suppress tokens

Chatterbox needs the talker sampler to apply the GGUF's
`tokenizer.ggml.suppress_tokens` as `-inf` logit biases, as upstream
`common_sampler` does; [#180](https://github.com/leehack/llamadart-native/pull/180)
adds that. Its converter suppresses every text token. Without the bias
Chatterbox draws them: with seeds 2, 3 and 5 its first token was GPT-2
`<|endoftext|>` (an end-of-generation token), which ends the utterance with no
audio. Upstream `llama-tts` gave 61 to 72 frames for seeds 1 to 8 with the same
settings.

The Chatterbox GGUFs embed upstream's defaults in `general.sampling.*`: Turbo
`top_k 1000`, `top_p 0.95`, `min_p 0`, `temp 0.8`; Multilingual `top_k 0`,
`top_p 1`, `min_p 0.05`, `temp 0.8`. Their `penalty_repeat 1.2` is inert
upstream: `penalty_last_n -1` clamps to 0. The wrapper's request defaults
(`top_k 40`, `top_p 0.95`, `min_p 0`, `temp 0.8`) also give intelligible
speech; callers may pass the GGUF values to match `llama-tts`.

## Why Pocket TTS is included

It needs no new wrapper machinery. Its helper pipeline uses the step contract
LFM3-Audio already exercises (no backbone token, frames from the hidden state,
`stop` from the helper), and its voices are the same voice bitmaps. The bit is
gated on the voice API (`mtmd_gen_audio_get_voice`), so a build on older
headers still reports it `UNKNOWN`. Long text is split into chunks inside the
helper, each restarting from the voice; frames of every chunk count towards
`max_frames`.

## Not exposed

- Streaming PCM and `mtmd_helper_gen_audio_flush`. Chatterbox's flow decode is
  one-shot anyway.
- `mtmd_gen_audio_read_tensor`, `mtmd_gen_state_set_ref` and raw voice bitmaps:
  the helper calls them; callers name voices.
- Chatterbox CFG strength (fixed at 0.5 upstream).

## Known upstream gaps

- Multilingual text is not normalized
  ([#31686](https://github.com/ggml-org/llama.cpp/issues/31686)): diacritics of
  some languages drop. The wrapper passes `request.text` as given and does not
  validate the language; check it against the model card.
- A short speaker clip makes Chatterbox run on: a 3 s clip reached 512 frames
  in both `llama-tts` and the wrapper, an 8.7 s clip ended at 67 to 84 frames.
  Upstream recommends about 10 s.
- The prompt is tokenized with special-token parsing (Chatterbox), as in
  `llama-tts`.

## Local check

`llamadart_tts_voice_smoke` (`-DLLAMADART_BUILD_TTS_SMOKE=ON`) takes either
generator:

```bash
build/tts-smoke/llamadart_tts_voice_smoke MODEL MMPROJ /tmp/voice-smoke \
  [--speaker REFERENCE.wav] [--gpu]
```

It checks that callers without the type's bit see `UNKNOWN`, the capability
metadata, voice listing (length query, short buffer, out-of-range index),
unknown-voice, voice-plus-reference, language and sequence rejections, the
Multilingual refusal of a one-sequence context, cancel before and during
generation and that both sequences are empty afterwards, that a request
without a voice equals one naming the first voice in upper case on another
sequence, that another voice, another seed and a reference each change the
PCM, a French Multilingual synthesis and `max_frames` truncation. It writes
`-default.wav`, `-<last voice>.wav`, `-cloned.wav` and `-fr.wav`.

Results on macOS arm64 (M4 Max) at `cd959bae`, CPU and Metal, all `PASS`;
transcripts from Qwen3-ASR-0.6B
([ggml-org/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF)
`928ab958`, Q8_0):

| Model (repository @ revision) | Files (sha256 prefix) | Transcript of `-default.wav` |
| --- | --- | --- |
| [ggml-org/chatterbox-turbo-GGUF](https://huggingface.co/ggml-org/chatterbox-turbo-GGUF) @ `66f03935` | `chatterbox-turbo-Q8_0.gguf` (`c3b93d3c`), `mmproj-chatterbox-turbo-F16.gguf` (`de86b2ea`) | "Hello from Mama Dart. How are you today?" (CPU), "Hello from Llama Dart. How are you today?" (Metal) |
| [ggml-org/chatterbox-GGUF](https://huggingface.co/ggml-org/chatterbox-GGUF) @ `a5c19d4a` | `chatterbox-Q8_0.gguf` (`8fb23a40`), `mmproj-chatterbox-F16.gguf` (`e2bc7506`) | "Hello from Llama Dart. How are you today?"; `-fr.wav`: "Bonjour, comment allez-vous aujourd'hui?" |
| [macolomb/pocket-tts-without-voice-cloning](https://huggingface.co/macolomb/pocket-tts-without-voice-cloning) @ `cc72b003` | `english/pocket-tts-english.gguf` (`7146564b`), `english/mmproj-pocket-tts-english.gguf` (`35cbf36b`) | "Hello from Llama Dart. How are you today?" (also voice `vera`) |

The ggml-org Multilingual card asks for a build with #31698; at v0.9.0 the
converted file still runs, detected by its position tables. The speaker
reference was 8.7 s of Pocket TTS voice `vera` from upstream `llama-tts`
(sha256 `1228cbef`); the cloned outputs transcribe to the same sentence, with
"Llama Dart" heard as "Lama Dart" or "llama dot" on Turbo.
The Kyutai packs ship no voice encoder, so Pocket TTS reference cloning is
covered only by its `UNSUPPORTED` refusal.

Regressions at the same commit, CPU and Metal: `llamadart_tts_smoke` with
Qwen3-TTS-12Hz-1.7B-Base Q4_K_M (`8d18c94a`) and its Q8_0 mmproj (`6fd65188`),
`llamadart_tts_lfm3_smoke` with LFM3-Audio-2B Q4_K_M (`9a773289`) and its Q8_0
mmproj (`2b6648b1`), and `llamadart_soprano_tts_smoke` with Soprano-1.1-80M
converted from [ekwek/Soprano-1.1-80M](https://huggingface.co/ekwek/Soprano-1.1-80M)
@ `27b5a5f5` by the v0.9.0 converter (`b02f1225`, mmproj `4fe5de0c`) all pass.
The Soprano mmproj conversion needed the converter's exact
`torch.equal` Hann-window check relaxed: the stored window differs from
`torch.hann_window` by 6e-8 under torch 2.11.
