# v0.8.0 qualification

Stable upstream v0.8.0, `8642541b255bfb5dc39cd351a178be09013a55d6`, is the
next runtime candidate (issue #112). It is nightly b11912 plus 48 commits and
v0.7.0 `620d2225cc436ab0b001e3200e590e3b286ef9ab` plus 291. Its native
release identity is the stable tag `v0.8.0`, which `latest` selects.

The production submodule stays at v0.7.0. `validate_wrapper.yml` builds v0.8.0
in a `v0.8.0` row of every native lane.

The scheduled stable dispatcher already selects v0.8.0. On 2026-10-29 it
dispatched `native_release.yml` from `main` before this change, and every
build job failed compiling the wrapper against the v0.8.0 `mtmd` API; nothing
was published. Once this change is on `main`, the next scheduled run can build
and publish `v0.8.0` without a further code change.

## Wrapper change

Upstream #30902 (first in b11912) replaces the projector's shared generation
RNG with an `mtmd_gen_state` owned by each generation: `mtmd_gen_audio_process`
gains a state argument and `mtmd_gen_inp` loses `seed`, `state_data` and
`state_size`. `mtmd_helper_gen_audio_set_input` seeds a fresh state from the
helper input's `seed`, which the wrapper already sets. The v0.7.0 priming call
from #122 is therefore compiled only when `LLAMADART_MTMD_HAS_GEN_STATE` is
false. The wrapper's exported API is unchanged: 89 required exports.

v0.8.0 adds `n_past`, `out_mode` and `voice` to `mtmd_helper_gen_audio_inp`.
The wrapper value-initialises that struct, so they stay off (no sequence
continuation, no voice). It still accepts only `MTMD_GEN_AUDIO_TYPE_QWEN3TTS`;
the new Soprano and LFM3-Audio generators report unsupported.

## Carried patches

All three apply strictly to v0.8.0.

- 0001 (Adreno 750, #79) is still needed: upstream PR #29165 is an open draft,
  and the matvec reduction-mode selection it guards is unchanged.
- 0002 (buffer device address, leehack/llamadart#782) is still needed: v0.8.0
  still takes `bufferDeviceAddress` from `VkPhysicalDeviceVulkan12Features`
  without checking for the entry point.
- 0003 (#114) is redundant on v0.8.0: upstream #31086 makes `set_embeddings`
  re-reserve on every toggle. A build without it passes every native test,
  including `model-session` and `model-embedding-toggle`. It stays in
  `series.json` while the pinned v0.7.0 and its rows still build from this
  series; remove it with the change that moves the pin to v0.8.0.

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
