# b11912 runtime candidate (0.17)

Upstream has no stable tag after v0.7.0, so the 0.17 runtime candidate is
nightly b11912, `961c376a2daa42220ec09663b9bde32c5fba1f70`, 243 commits after
v0.7.0 `620d2225cc436ab0b001e3200e590e3b286ef9ab` (issue #112). It is the
newest nightly whose only red upstream CI lane is riscv64, which this
repository does not build; b11913 adds only a Hexagon option and has a failing
Snapdragon device run.

The production submodule stays at v0.7.0. `validate_wrapper.yml` builds b11912
in a `b11912` row of every native lane. Under the release policy a b11912
artifact would be the explicit nightly `b11912`, a GitHub prerelease that
`latest` never selects. If upstream publishes a stable release that contains
this source first, qualify and ship that tag instead.

## Wrapper change

Upstream #30902 replaces the projector's shared generation RNG with an
`mtmd_gen_state` owned by each generation: `mtmd_gen_audio_process` gains a
state argument and `mtmd_gen_inp` loses `seed`, `state_data` and `state_size`.
`mtmd_helper_gen_audio_set_input` seeds a fresh state from the helper input's
`seed`, which the wrapper already sets. The v0.7.0 priming call from #122 is
therefore compiled only when `LLAMADART_MTMD_HAS_GEN_STATE` is false. The
wrapper's exported API is unchanged: 88 required exports.

`tools/tts_smoke.cpp` with Qwen3-TTS 12Hz 1.7B Base Q4_K_M
(SHA-256 `8d18c94acb2addd042f97da63c98be144eafa76d0d9495177eab65130cf85129`) and its Q8_0 projector (`6fd65188839bcd6ecc91b277ad471e22a0edfada4699a0fe82f1165c18cfcce2`) passed on
macOS arm64 CPU against b11912. A repeated seed repeats the PCM, distinct seeds
reach the code predictor, and the cancel checks pass.

## Carried patches

All three carries apply strictly to b11912.

- 0001 (Adreno 750) and 0002 (buffer device address) are still absent
  upstream; 0001's upstream PR #29165 is open.
- 0003 is redundant on b11912: #31086 makes `set_embeddings` re-reserve on
  every toggle. Applying it adds a second, identical assignment. Retire it
  only after the patchless tests recorded in #114 pass.

## Consumer ABI delta

`llama.h` and `ggml-backend.h` declare no new or changed function, struct or
enum; `llama.h` adds only `LLAMA_DRY_MAX_SEQ_BREAKERS` (64 distinct DRY
breakers; later ones are ignored with a warning). No public symbol is removed.
Regenerated bindings must account for:

| Header | Change |
| --- | --- |
| `ggml.h` | `GGML_OP_TOP_K_MASK` is inserted before `GGML_OP_UNARY`, so `GGML_OP_UNARY` and every later `ggml_op` value, including `GGML_OP_COUNT`, increase by 1. Adds `ggml_top_k_mask`, `ggml_prec_get_acc`, `ggml_prec_get_src`. |
| `mtmd.h` | `mtmd_gen_audio_process(ctx, state, inp, out)` gains a state argument. `mtmd_gen_inp` drops `seed`, `state_data`, `state_size`; `mtmd_gen_out` drops `state_data`, `state_size`. `mtmd_caps` adds `gen_audio`. `mtmd_gen_audio_type` appends `MTMD_GEN_AUDIO_TYPE_SOPRANO`. Adds `mtmd_gen_state_*` (4 functions, `mtmd_gen_state_params`), `mtmd_support_audio_stream` and the experimental `mtmd_audio_stream_*` API (9 functions, `mtmd_audio_stream_params`). |
| `mtmd-helper.h` | `mtmd_helper_gen_audio_inp` gains a leading `bool stream`, shifting every field. Adds `mtmd_helper_get_capabilities` (`mtmd_helper_cap`), `mtmd_helper_gen_audio_inp_default`, `mtmd_helper_gen_audio_flush`. `mtmd_helper_model_can_chat` and `mtmd_helper_support_video` are deprecated but still exported. |

## Remaining gates

- Done: the hosted `b11912` rows of every lane in `validate_wrapper.yml` pass,
  and the non-publishing `native_release.yml` artifact preparation run
  [succeeded](https://github.com/leehack/llamadart-native/actions/runs/40474126491).
- Android ISA: [b11912_android_isa_qualification.md](b11912_android_isa_qualification.md).
- Model, family and device qualification stays under #112 and the consumer
  release; this document claims none of it.
