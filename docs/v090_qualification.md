# v0.9.0 qualification

Stable upstream v0.9.0, `cd959bae99ff4bed22bef2604f45241c09bbb8c8`
(2026-11-04), is the next runtime candidate. It is v0.8.0
`8642541b255bfb5dc39cd351a178be09013a55d6` plus 150 commits. Its native release
identity is the stable tag `v0.9.0`, which `latest` selects.

This change moves the production submodule to v0.9.0 and adds a `v0.9.0` row
to every lane in `validate_wrapper.yml`. Native `v0.9.0` is not published:
the 2026-11-05 automatic dispatch failed at the CPU ISA audits
([v090_android_isa_qualification.md](v090_android_isa_qualification.md)), and
the next scheduled dispatch after this change reaches `main` publishes it.

## Wrapper

The wrapper builds against v0.9.0 without source changes, and every
`CMakeLists.txt` header probe succeeds. The exported API is unchanged.

- `mtmd_gen_audio_info` gains `n_voices`. The wrapper reads only `type` and
  `sample_rate`.
- `mtmd_helper_gen_audio_inp` loses `voice`, which the wrapper never set.
- `MTMD_GEN_AUDIO_TYPE_CHATTERBOX` maps to `LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN`,
  so the TTS entry points report it unsupported. Chatterbox and the new
  predefined-voice API are not wrapped.

Evidence, 2026-11-06, Apple M4 Max, macOS arm64, Metal on, Release,
`LLAMADART_BUILD_TESTS=ON`, `LLAMADART_BUILD_TTS_SMOKE=ON`: native CTest
84/84; `tools/validate_exports.py` with `--forbid-import __cxa_atexit` passes
(92 required exports); `tools/llama_cpp_patches.py check-apple` passes on
`libllamadart.dylib`.

## Carried patches

`tools/llama_cpp_patches.py apply --strict` succeeds on v0.9.0 and applies only
0001 and 0002. `manifest --upstream` lists only those two, so a v0.9.0
`assets.json` lists only them. `series.json` is unchanged.

- 0001 (Adreno 750, #79) is still needed: upstream PR #29165 is still an open
  draft, and the matvec reduction selection it guards is unchanged.
- 0002 (buffer device address, leehack/llamadart#782) is still needed: v0.9.0
  still takes `bufferDeviceAddress` from `VkPhysicalDeviceVulkan12Features`
  without checking for the entry point.
- 0003 (#114) stays retired, as on v0.8.0: its `upstream_fix` is in
  `src/llama-context.cpp`.
- 0004 (#154) is retired on v0.9.0: upstream #31360 (`c9daee1`) is in v0.9.0,
  and `unary.metal` contains its `upstream_fix` text. The patch is not applied
  and `check-apple` does not require its marker. Its `required_text`, the
  v0.8.0 `inline constexpr` form, no longer occurs in v0.9.0.
  `metal_os_floor.yml` checks the unpatched v0.9.0 kernels on the oldest
  runners.

Delete 0004 once no `validate_wrapper.yml` row builds v0.8.0, and 0003 once
none builds v0.7.0 or older.

## Consumer ABI delta from v0.8.0

No public C symbol is removed. Regenerated bindings must account for:

| Header | Change |
| --- | --- |
| `llama.h` | Adds `LLAMA_FTYPE_MOSTLY_F8_E4M3 = 42`. |
| `ggml.h` | Adds `ggml_fp8_e4m3_t`, `GGML_TYPE_F8_E4M3 = 43` (`GGML_TYPE_COUNT` becomes 44), `GGML_FTYPE_MOSTLY_F8_E4M3 = 29` and `ggml_flash_attn_ext_rows`. |
| `ggml-backend.h` | Adds `ggml_backend_tensor_copy_n`, `ggml_backend_tensor_set_n`, `ggml_backend_tensor_get_n`, and the `ggml_backend_comm_bootstrap_callback` and `ggml_backend_comm_init_rank_t` typedefs. |
| `mtmd.h` | `mtmd_gen_audio_info`, returned by value, appends `int32_t n_voices` and grows from 16 to 24 bytes on 64-bit targets. Appends `MTMD_VIDEO_TEMPLATE_QWEN3VL` and `MTMD_GEN_AUDIO_TYPE_CHATTERBOX`. Adds `mtmd_bitmap_is_voice`, `mtmd_gen_audio_read_tensor`, `mtmd_gen_audio_voice_name`, `mtmd_gen_audio_get_voice` and `mtmd_gen_state_set_ref`. |
| `mtmd-helper.h` | `mtmd_helper_gen_audio_inp` drops `voice` after `lang`, so every later field moves back 8 bytes and the struct shrinks from 72 to 64 bytes on 64-bit targets. The C++ `gen_audio::inp` drops `set_voice`. Adds the C++-only test types `mtmd_helper_video_src` and `mtmd_helper_video_seq`. |

`common.h`, which the bindings do not include, drops `fs_path_to_utf8` and
gives `common_control_vector_load` an `n_layer` argument; the wrapper uses
neither.

## Remaining gates

- Hosted `v0.9.0` and `pinned` rows of every lane in `validate_wrapper.yml`,
  `validate_linux_release_rows.yml` and `metal_os_floor.yml`.
- Android ISA: [v090_android_isa_qualification.md](v090_android_isa_qualification.md).
- Model, family and device qualification stays with the consumer release; this
  document claims none of it.
