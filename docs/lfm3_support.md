# LFM3 support

LFM3 ships in upstream stable v0.8.0 (`8642541b`) as one family:
text and VL ([#31252](https://github.com/ggml-org/llama.cpp/pull/31252)),
Audio and Omni ([#31257](https://github.com/ggml-org/llama.cpp/pull/31257)),
d1 decision models ([#31275](https://github.com/ggml-org/llama.cpp/pull/31275))
and video input ([#31283](https://github.com/ggml-org/llama.cpp/pull/31283)).
v0.7.0, the pinned submodule, has none of it. This page is the owner scope of
issue #130; it claims no model, device or consumer qualification.

## What needs a wrapper change

| Variant | Upstream v0.8.0 | Wrapper |
| --- | --- | --- |
| Text (LFM3-1.7B) | `LLM_ARCH_LFM3` in libllama | None: the generic load, decode and sampling exports. |
| VL (LFM3-VL-2B) | `lfm3vl` projector in mtmd | None: the generic `llama_dart_mtmd_` exports. The image-token limit stays Gemma 4 only. |
| Audio input, ASR (LFM3-Audio, LFM3-Omni) | `lfm3a` encoder in mtmd | None: the generic `llama_dart_mtmd_` exports. |
| Speech output (LFM3-Audio, LFM3-Omni) | `MTMD_GEN_AUDIO_TYPE_LFM3A` | Added, opt-in: complete PCM only. |
| d1 (d1-2B, d1-Omni-2B) | `LLM_ARCH_LFM3`, non-causal, `lfm3-d1` and `lfm3-d1-omni` decision types | None: decision scoring is the consumer's. |
| Video (VL, d1, Omni soundtrack) | `mtmd_helper_video_*`, needs the `ffmpeg` binary | Not wrapped. |

d1: llama.cpp creates no memory for a non-causal LFM3 model, so one batch must
hold the whole prompt and there is no sequence state to keep, copy or save.

## Speech output

Old callers see no change: `llama_dart_tts_get_info` and `llama_dart_tts_init`
accept Qwen3-TTS only and report LFM3-Audio as `UNKNOWN` and `UNSUPPORTED`.
A caller opts in with `llama_dart_tts_get_info_ext` and
`llama_dart_tts_init_ext`, whose mask names
`LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO`. `llama_dart_tts_supported_model_types`
is the build probe: a wrapper built on v0.7.0 reports Qwen3 only, and an older
wrapper lacks the symbols.

The generator differs from Qwen3-TTS in ways an allowlist entry cannot cover:

- Its prompt must be the model's `tts` chat template, ending in
  `<|audio_start|>`. The wrapper renders `request.text` with llama.cpp's
  `common_chat_get_tts_prompt`, `request.language` and the voice.
- Frames come from the hidden state. The text head is not sampled, so a text
  end-of-generation token cannot end the speech; the end-of-audio code does.
- Upstream ignores a speaker reference with a warning; the wrapper refuses it.
- Codes are sampled with the GGUF's sampling settings; the request's `top_k`,
  `top_p`, `min_p` and `temperature` are unused. `seed` seeds the code sampler
  and the generation state.
- Each frame is decoded to PCM inside its step, so a cancel lands within one
  frame and there is no long final decode.

`llamadart_tts_lfm3_smoke` is the local check. With
[LiquidAI/LFM3-Audio-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Audio-2B-GGUF)
at `02355c1a`, `LFM3-Audio-2B-Q4_K_M.gguf` (sha256 `9a773289…5be2`) and
`mmproj-LFM3-Audio-2B-Q8_0.gguf` (`2b6648b1…e365`) pass on macOS arm64 CPU and
Metal, and ASR transcribes the speech back to its text.

## Not exposed

Each needs its own contract, not a flag on the TTS task:

- Streaming PCM (`stream`, `mtmd_helper_gen_audio_flush`).
- Omni mixed text and audio, and sequence continuation (`n_past`, `out_mode`,
  `mtmd_helper_gen_audio_next_is_text`, `mtmd_helper_gen_audio_get_logit_bias`):
  the caller samples the text slots.
- Live ASR (`mtmd_audio_stream_*`, experimental upstream).
- Video input. It runs `ffmpeg` as a subprocess, which mobile platforms lack.

The `tts` prompt is tokenized with special-token parsing, so control tokens in
`request.text` are parsed as tokens, as in upstream `llama-tts`.
