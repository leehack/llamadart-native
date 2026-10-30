# v0.8.0 qualification

Stable upstream v0.8.0, `8642541b255bfb5dc39cd351a178be09013a55d6`, is the
next runtime candidate (issue #112). It is nightly b11912 plus 48 commits and
v0.7.0 `620d2225cc436ab0b001e3200e590e3b286ef9ab` plus 291. Its native
release identity is the stable tag `v0.8.0`, which `latest` selects.

Native `v0.8.0` was published on 2026-10-30 and the production submodule is
at v0.8.0 (`update-llama-submodule`, 8005b57). `validate_wrapper.yml` keeps a
`v0.7.0` row of every native lane, so patch 0003 stays until that row goes.

The first scheduled dispatch, on 2026-10-29, failed in every build job
compiling the wrapper against the v0.8.0 `mtmd` API and published nothing; the
2026-10-30 dispatch built and published it.

## Wrapper change

Upstream #30902 (first in b11912) replaces the projector's shared generation
RNG with an `mtmd_gen_state` owned by each generation: `mtmd_gen_audio_process`
gains a state argument and `mtmd_gen_inp` loses `seed`, `state_data` and
`state_size`. `mtmd_helper_gen_audio_set_input` seeds a fresh state from the
helper input's `seed`, which the wrapper already sets. The v0.7.0 priming call
from #122 is therefore compiled only when `LLAMADART_MTMD_HAS_GEN_STATE` is
false. This change leaves the wrapper's exported API unchanged.
`DEFAULT_REQUIRED_SYMBOLS` in `tools/validate_exports.py` holds the current
list, which later opt-in exports extend.

v0.8.0 adds `n_past`, `out_mode` and `voice` to `mtmd_helper_gen_audio_inp`.
The wrapper value-initialises that struct, so they stay off (no sequence
continuation, no voice). `llama_dart_tts_get_info` and `llama_dart_tts_init`
still accept only `MTMD_GEN_AUDIO_TYPE_QWEN3TTS`. LFM3-Audio is opt-in through
their `_ext` variants ([lfm3_support.md](lfm3_support.md)), and so is Soprano
(README, "Soprano").

## Carried patches

All three apply strictly to v0.8.0; 0001 and 0002 are applied.

- 0001 (Adreno 750, #79) is still needed: upstream PR #29165 is an open draft,
  and the matvec reduction-mode selection it guards is unchanged.
- 0002 (buffer device address, leehack/llamadart#782) is still needed: v0.8.0
  still takes `bufferDeviceAddress` from `VkPhysicalDeviceVulkan12Features`
  without checking for the entry point.
- 0003 (#114) is retired on v0.8.0: upstream #31086 (`74bca395`, first in
  b11870) makes `set_embeddings` re-reserve on every toggle. Its
  `upstream_fix` text is that condition, so a v0.8.0 build does not apply it,
  its Android marker is not required and `assets.json` does not list it.
  v0.7.0 lacks the text and still applies it. Delete the patch once no
  `validate_wrapper.yml` row builds v0.7.0 or older.

  Evidence, 2026-10-29, Apple M4 Max, macOS arm64, CPU/Accelerate, Metal and
  KleidiAI off, Release:

  | Build | `model-session` | `model-embedding-toggle` | Native CTest |
  | --- | --- | --- | --- |
  | v0.8.0 with 0003 | pass | pass | 57/57 |
  | v0.8.0 without 0003 | pass | pass | 57/57 |
  | v0.7.0 without 0003 (negative control) | SEGFAULT | abort | not run |

  Each test passes on the same v0.8.0 builds with real SmolLM2-135M-Instruct
  Q8_0 and Qwen3.5-0.8B Q4_K_M weights, and exits 139 with both on the v0.7.0
  negative control.

## Consumer ABI delta from v0.7.0

`llama.h` adds only `LLAMA_DRY_MAX_SEQ_BREAKERS` (64 distinct DRY breakers;
later ones are ignored with a warning). `ggml-backend.h` is unchanged. No
public symbol is removed. Regenerated bindings must account for:

| Header | Change |
| --- | --- |
| `ggml.h` | `GGML_OP_TOP_K_MASK` is inserted before `GGML_OP_UNARY`, so `GGML_OP_UNARY` and every later `ggml_op` value, including `GGML_OP_COUNT`, increase by 1. Adds `ggml_top_k_mask`, `ggml_prec_get_acc`, `ggml_prec_get_src`. |
| `mtmd.h` | `mtmd_gen_audio_process(ctx, state, inp, out)` gains a state argument. `mtmd_gen_inp` drops `seed`, `state_data`, `state_size` and inserts `penalty_repeat`, `penalty_last_n` after `temp`; `mtmd_gen_out` drops `state_data`, `state_size`. `mtmd_caps` adds `gen_audio`. `mtmd_gen_audio_type` appends `SOPRANO` and `LFM3A`. `mtmd_audio_stream_params` appends `prompt`, `n_prompt`. Adds `mtmd_gen_state_*` (4 functions, `mtmd_gen_state_params`), `mtmd_support_audio_stream`, the experimental `mtmd_audio_stream_*` API (10 functions), `mtmd_video_template`, `mtmd_video_config`, `mtmd_get_video_config` and `mtmd_bitmap_set_is_video`. |
| `mtmd-helper.h` | `mtmd_helper_gen_audio_inp` gains a leading `bool stream`, then `n_past` and `out_mode` after `prompt_len` and `voice` after `lang`, shifting every later field. `mtmd_helper_video_init_params` appends `no_soundtrack`. Adds `mtmd_helper_get_capabilities` (`mtmd_helper_cap`), `mtmd_helper_gen_audio_inp_default`, `mtmd_helper_gen_audio_flush`, `mtmd_helper_gen_audio_next_is_text`, `mtmd_helper_gen_audio_get_logit_bias`, `mtmd_helper_audio_stream_get_logit_bias` and `mtmd_helper_gen_audio_out_mode`. `mtmd_helper_model_can_chat` and `mtmd_helper_support_video` are deprecated but still exported. |

## Remaining gates

- Hosted `v0.8.0` rows of every lane in `validate_wrapper.yml`.
- Android ISA: [v080_android_isa_qualification.md](v080_android_isa_qualification.md).
- Model, family and device qualification stays under #112 and the consumer
  release; this document claims none of it.
