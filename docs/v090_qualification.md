# v0.9.0 qualification

Stable upstream v0.9.0, `cd959bae99ff4bed22bef2604f45241c09bbb8c8`
(2026-11-04), is the next runtime candidate. It is v0.8.0
`8642541b255bfb5dc39cd351a178be09013a55d6` plus 150 commits. Its native release
identity is the stable tag `v0.9.0`, which `latest` selects.

This change moves the production submodule to v0.9.0 and adds it as the
pinned `LLAMADART_UPSTREAMS` row of `validate_wrapper.yml`, next to the v0.7.0
and v0.8.0 rows. Native `v0.9.0` is not published: the 2026-11-05 automatic
dispatch failed at the CPU ISA audits
([v090_android_isa_qualification.md](v090_android_isa_qualification.md)).
Merging this change publishes nothing; `auto_native_release.yml` runs only on
manual dispatch.

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
`libllamadart.dylib`. With patches 0005 and 0006 (2026-11-07, preset
`macos-arm64-full`, `LLAMADART_BUILD_TESTS=ON`): CTest 85/85, including
`llamadart_f16_dot_accumulation_test`; the same export and `check-apple`
checks pass.

## Carried patches

Reviewed 2026-11-07 against v0.9.0 and upstream `master`
`f7591a6a01f9273de03331732bdc0b39721168be` (b12190).
`tools/llama_cpp_patches.py apply --strict` succeeds on both and applies 0001,
0002, 0005 and 0006, the four that `manifest --upstream` lists. Every patch also
passes `git apply --check` on both. `series.json` is unchanged.

| Patch | v0.9.0 | Upstream | Status |
| --- | --- | --- | --- |
| 0001 Adreno 750 matvec (#79) | Applied | [#29165](https://github.com/ggml-org/llama.cpp/pull/29165) open draft; `use_subgroups` selection unchanged on `master` | Keep |
| 0002 buffer device address (leehack/llamadart#782) | Applied | No fix: `master` still takes `bufferDeviceAddress` from `VkPhysicalDeviceVulkan12Features` without checking the entry point; upstream supports Vulkan 1.2 and later only | Keep |
| 0003 embedding-mode reserve (#114) | Not applied: `upstream_fix` present ([#31086](https://github.com/ggml-org/llama.cpp/pull/31086)) | Fixed since v0.8.0 | Needed only by the v0.7.0 row |
| 0004 Metal unary cast (#154) | Not applied: `upstream_fix` present, `required_text` absent ([#31360](https://github.com/ggml-org/llama.cpp/pull/31360)) | Fixed after v0.8.0 | Needed only by the v0.8.0 row |
| 0005 ARM F16 dot (#174) | Applied | [#31734](https://github.com/ggml-org/llama.cpp/issues/31734) open, no fix; non-SVE NEON `ggml_vec_dot_f16` still accumulates in `float16x8_t` | Keep |
| 0006 ARM F16 GEMM (#174) | Applied | Same issue; llamafile still uses `tinyBLAS<8, float16x8_t, …>` | Keep |

v0.7.0 needs 0001–0003, 0005 and 0006 (0004 does not fit it, since it predates
the regression). v0.8.0 needs 0001, 0002 and 0004–0006. Deleting 0003 means
dropping the v0.7.0 `LLAMADART_UPSTREAMS` row (cores 0.15–0.17); deleting 0004
means dropping the v0.8.0 row (core 0.18). Both are support-policy decisions
for the maintainer.

`check-apple` does not require the 0004 marker on v0.9.0, and
`metal_os_floor.yml` checks the unpatched v0.9.0 Metal kernels on the oldest
runners.

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

- Hosted `v0.9.0` rows of every lane in `validate_wrapper.yml`, and
  `validate_linux_release_rows.yml` and `metal_os_floor.yml` at the pin.
- Android ISA: [v090_android_isa_qualification.md](v090_android_isa_qualification.md).
- Model, family and device qualification stays with the consumer release; this
  document claims none of it.
