# v1.1.0 native candidate

Owner: [native #239](https://github.com/leehack/llamadart-native/issues/239),
coordinated by [core #2337](https://github.com/leehack/llamadart/issues/2337).
This is an experimental native-source candidate, not a qualified release.
Proposed artifact version **v1.1.0** follows
[the stable-version policy](release_version_policy.md). Merge, preparation,
hardware/model execution, publication and consumer adoption have separate
gates.

| Input | Exact revision |
| --- | --- |
| Owner base | `36d7bb31786049419a4fe5d28aebf2565ed41951` ([#234](https://github.com/leehack/llamadart-native/pull/234)) |
| Baseline llama.cpp | `b1b770d90d94cad9e9ea5b993bdbc383a522fb28` (v1.0.0) |
| Fixed candidate | `171527522610996f5f0acf12334e185156ca8f44` |
| Stable annotated tag object | `bdda03620c7355b4779afa8f78cb86129e9bf408` |

The dereferenced published `v1.1.0` tag supplies the candidate. The release
object's `target_commitish=074f1f3f…` is a different commit and is not used.
No nightly/main changes, new speech admissions, cache types or public controls
are part of this candidate: the new upstream audio types (Zonos3, YuE2,
MiniMax-Music3, Fish S2, Moshi, Kokoro, tiny-voca, VoxCPM2) stay
`LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN` and keep their existing refusals, tracked in
[#189](https://github.com/leehack/llamadart-native/issues/189),
[#195](https://github.com/leehack/llamadart-native/issues/195),
[#210](https://github.com/leehack/llamadart-native/issues/210),
[#219](https://github.com/leehack/llamadart-native/issues/219) and
[#227](https://github.com/leehack/llamadart-native/issues/227). Native CI
qualifies only v1.1.0, the pinned upstream.

## Wrapper adaptations

| Upstream change | Adaptation | Control |
| --- | --- | --- |
| `mtmd_helper_gen_audio_set_input` returns 2 for `speed <= 0`; the wrapper's zero-initialized input made every TTS setup fail with "TTS input setup failed". | `llama_dart_audio_input_compat` sets `speed = 1.0f` when `mtmd_helper_gen_audio_inp::speed` exists (probe `LLAMADART_MTMD_AUDIO_HAS_SPEED`). `loudness`, `cfg_scale` and `flow_steps` default to 0 (the model's own) and stay zero. | `mtmd_compat_test` asserts speed 1.0 and the upstream defaults for every model type and Pocket variant; it failed before the change. |
| `mtmd_helper_gen_audio_step_prompt` returns -1 when its batch argument exceeds `llama_n_batch()` of the context. The wrapper passed the request's `prompt_batch_size` (default 512) unchanged, so TTS failed with "TTS prompt processing failed" on any context with `n_batch` below it (Pocket TTS, `n_batch` 256; v1.0.0 passed). | The step clamps to `min(prompt_batch_size, llama_n_batch(ctx))` (`llama_dart_tts_prompt_batch`). This also avoids v1.0.0's assertion for a prompt longer than `n_batch`. | `llamadart_tts_batch_test`, red before the clamp and killed by dropping it. Real Pocket TTS english model, `n_ctx` 2048, `prompt_batch_size` 512: `n_batch` 256 and 2048 both generate 6 frames / 11520 samples. |
| `ggml-hip` requires ROCm/HIP 6.3 or newer (v1.0.0: 6.1). | The Linux HIP release and validation containers move from `rocm/dev-ubuntu-22.04:6.1.2` to `6.3.4` (same Ubuntu 22.04 base, so the glibc floor is unchanged). | `build-linux-hip` in `validate_linux_release_rows.yml`; `test_native_release_dedup.py`. **Consumer-facing:** the shipped `libggml-hip.so` is now built against ROCm 6.3.4, so the ROCm runtime requirement is 6.3 or newer for the first time; the exact soname dependencies are checked by the glibc and artifact contracts and must be recorded in the consumer's release notes. |
| `draft-dflash2` and `draft-lilicorr` split out of DFlash ([#32016](https://github.com/ggml-org/llama.cpp/pull/32016)). | Both count as block drafts for the draft-context output limits and as external draft contexts (`llama_dart_type_mask_has_draft_context`, `..._non_mtp_draft_context`, `llama_dart_count_draft_context_types`, `llama_dart_apply_speculative_draft_output_limits`), moved into `llama_dart_speculative_compat.h` so the test can call them. Probe `LLAMADART_SPECULATIVE_HAS_DFLASH2`. | `speculative_api_test` rows for both types and the mask helpers; removing the two types from the block-draft predicate fails the output-limit rows. |
| Both types are inserted before the n-gram types in `common_speculative_type`. | None. `type_mask` bit positions of the n-gram types move by two; callers keep using `type_names`, as the header already documents. Bits 0 to 5 (through `draft-dspark`) keep their value. | Mask helpers are tested by type, not by literal bit. |
| `ggml-cuda.cu` includes `<nvml.h>` in every NVIDIA build. | `cuda_nvml_dev` joins the default and required components of `tools/install_cuda_redist.py`. It is in CUDA 12.8.1's redistrib manifest for `linux-x86_64` and `windows-x86_64` (12.8.90, `include/nvml.h`, SHA-256 verified). The library is loaded dynamically; no import library is linked. | `tests/test_install_cuda_redist.py`; the Linux CUDA row of `validate_linux_release_rows.yml` and the Windows CUDA release lane compile it. |
| `GGML_OPENCL_USE_ADRENO_BIN_KERNELS` defaults ON and loads `libadreno-opencl-kernels.so`. | Android configure arguments set it OFF, which is v1.0.0's behavior; no bundle ships that library. | `tests/test_android_opencl_options.py`. |
| `llama_batch`, `llama_batch_get_one`, `llama_encode`, `llama_decode` are deprecated for `llama_batch_ext` and `llama_process` ([#32306](https://github.com/ggml-org/llama.cpp/pull/32306)). | Unchanged: the wrapper's decode path still uses them and they remain exported. Migration is tracked separately. | ABI canary pins the `llama_batch` layout and resolves each deprecated export. |

## Carry decisions

Removing a patch still needed by the pinned upstream requires original-trigger
evidence of **old unpatched FAIL / old patched PASS / candidate unpatched
PASS**, with other fixes held constant. Application, marker presence, issue
closure and a successful build cannot replace that evidence.
[series.json](../patches/llama.cpp/series.json) lists the current carries. All
seven patches of v1.0.0 applied strictly to v1.1.0 before 0001 was removed.

| Carry | Candidate decision and proof/gap |
| --- | --- |
| 0001, [native #79](https://github.com/leehack/llamadart-native/issues/79) | **Retired.** The original S24 trigger failed on old unpatched, passed on old patched, and passed on candidate unpatched ([receipts](#hardware-and-model-receipts)). [Upstream #30826](https://github.com/ggml-org/llama.cpp/pull/30826) (`8bf88ba9`, in v1.1.0) disables subgroup arithmetic on the Qualcomm proprietary driver, including the subgroup-reduction dequant `mul_mat_vec` shaders whose pipeline creation crashed on the original trigger. The release-glslc rerun confirmed Qwen Q4_0 parity with v1.0.0-4. |
| 0002, [core #782](https://github.com/leehack/llamadart/issues/782) | Retain. The null `vkGetBufferDeviceAddress` guard is still absent upstream. Same gaps as [v1.0.0](v100_qualification.md): null/present-entry causal controls and native reachability on SC-53C/API36 Mali-G68. |
| 0005, 0006, [native #174](https://github.com/leehack/llamadart-native/issues/174) | Retain. The old F16 dot and FP16 llamafile GEMM kernels remain. Their model-free direct-kernel controls are unchanged and pass on the candidate CPU library. |
| 0007, [core #2034](https://github.com/leehack/llamadart/issues/2034) | Retain. `common/fit.cpp` still holds the static fit-log state; the strict application and the `fit_api_test` pass. The `upstream` permalink in `series.json` stays on the v1.0.0 lines it describes. |
| 0008, 0009, [native #192](https://github.com/leehack/llamadart-native/issues/192) | Retain. The unhinted F16 x F32 `GGML_OP_MUL_MAT` still takes upstream's F32 route. Re-verified against [#30846](https://github.com/ggml-org/llama.cpp/pull/30846) (`660be550`): the patched `ggml_compute_forward_mul_mat_one_chunk` selects `ggml_vec_dot_f16` and `GGML_TYPE_F16` before the new blocked loop reads `vec_dot_type`, `row_size`, `wdata` and `src1_col_stride`, so the converted-F16 work buffer layout is used consistently. The new tiled path (`ggml_compute_forward_mul_mat_tiled`) dispatches K-quant types only and returns false for F16. Per-patch removal needs the #192 throughput gate below. |

0003 and the Metal unary cast workaround were removed earlier and are not
restored.

## ABI and used-export handoff

`llamadart_upstream_abi_test` checks defaults only after scalar build-identity
admission, as in [v1.0.0](v100_qualification.md#abi-and-used-export-handoff).
The negative old/missing/unknown/dirty/duplicate-source controls make **zero**
by-value calls. Compiled macOS arm64 canaries:

| Contract | v1.0.0 | v1.1.0 |
| --- | --- | --- |
| `llama_context_params` / `llama_model_params` / `llama_batch` size | 176 / 80 / 56 | 176 / 80 / 56 |
| `llama_model_quantize_params` size | 64 | 136 |
| `mtmd_helper_gen_audio_inp` size | 72 | 96; `speaker_ref_text` inserted, so `lang` 40→48, `top_k` 48→56, `temp` 56→64, `seed` 60→68, `out_type` 64→88; `speed` 72, default 1.0 |
| `mtmd_gen_audio_info` size | 24 | 32; `n_channels` at 8, `model_variant` 8→16, `n_voices` 16→24 |
| `mtmd_gen_inp` size | 72 | 112 |
| `mtmd_helper_init_opt` / `mtmd_helper_video_init_params` size | 32 / 32 | 40 / 40 (`decoder` added) |
| `mtmd_helper_cap` size / `text_gen` offset | 8 / 5 | 9 / 6 (`audio_duplex` at 5) |
| `ggml_backend_dev_caps` size | 5 | 6 (`fast_async_host_input` at 5) |
| `GGML_OP_COUNT` | 99 | 101 |
| Optional RPC wire version | 10.0.0 | 10.0.2; match peers before use |

Struct layouts are consumed by value by the consumer's generated bindings:
the consumer must gate every by-value call on the build-identity admission,
including the new `mtmd_helper_gen_audio_inp` and `mtmd_gen_inp` sizes.
`llama_dart_build_identity()` is object-free and identifies the exact upstream.

The required-export validation passes **113** exports with no `__cxa_atexit`
import. The deprecated `llama_batch_*`, `llama_encode` and `llama_decode`
entry points are present. A full consumer used-symbol inventory against v1.1.0
headers belongs to the consumer's header and binding regeneration
([core #2337](https://github.com/leehack/llamadart/issues/2337)).

## Integration protections

The eighteen protections I01 to I18 of [v1.0.0](v100_qualification.md#integration-protections)
remain. Rows whose upstream area changed need their remaining qualification
repeated against v1.1.0:

| ID | v1.1.0 delta |
| --- | --- |
| I10, I14 | Audio output gains `n_channels` (interleaved) and the TTS pipelines gain delivery controls. The wrapper still admits only mono families and rejects every other model type. Re-run the media callback/cancellation and PCM-parity packs below. |
| I12 | `draft-dflash` is now four draft methods and GDN uses replay instead of rollback ([#31669](https://github.com/ggml-org/llama.cpp/pull/31669)). Block-draft output sizing covers all four. The MTP rollback probe and fresh-target contract pass on the generated components; a genuine recurrent MTP/DFlash pair is required before the rollback path is qualified. |
| I17 | ABI canaries updated, above. |

## Local checks

Run on macOS arm64 (AppleClang), 2026-11-26, from the candidate head. CPU-only
generated fixtures, no pretrained model.

```sh
cmake -S . -B <scratch>/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLAMADART_BUILD_TESTS=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DGGML_OPENMP=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16
cmake --build <scratch>/build -j 8
ctest --test-dir <scratch>/build --timeout 45 --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/llama_cpp_patches.py apply --strict --upstream third_party/llama.cpp --output <scratch>/strict
python3 tools/validate_exports.py --format nm --tool nm --forbid-import __cxa_atexit <scratch>/build/libllamadart.dylib
python3 scripts/verify_release_provenance.py
```

Results: **101/101** CTest tests pass, including `llamadart_upstream_abi_test`
and `llamadart_build_identity_test`; **515** Python tests pass with one
host-only skip; strict series application passes; 113 required exports are
present with no `__cxa_atexit` import; release-provenance contracts pass;
`actionlint` reports only the installed linter's existing VS2026 runner-label
exception. Before the fixes, `llamadart_mtmd_compat_test` failed its speed
assertion, the speculative output-limit rows failed with the DFlash2/LiliCorr
types removed from the block-draft predicate, the CUDA component test failed
without `cuda_nvml_dev`, the OpenCL option test failed without the Android
flag, and the unchanged Android ISA validator rejected the candidate module as
an unaudited source pair. These are component controls, not platform or model
qualification.

## Hardware and model receipts

2026-11-26, owner `b56515d6` (the merged candidate). Cloud runs used the maintainer's personal project; every VM, disk and storage object was deleted afterwards.

| Gate | Host | Result |
| --- | --- | --- |
| 1 Hosted PR lanes | GitHub Actions | 28 pass, 1 skipped (`macos-apple-xcframework-tools`) at the merged head. |
| 2 Retirement of 0001 | Firebase Test Lab SC-51E/API36, Adreno 750, driver `2150604839`, build `c81b28e1a5`, stories15M F32 and Qwen3.5-0.8B Q4_0 | **Pass.** Old unpatched (v1.0.0 source without 0001, NDK glslc 2022.3) hit `SIGSEGV` at `0x2d0` in `CreateQGLCProgram` under `vkCreateComputePipelines` on both models. Old patched (v1.0.0-4, CI glslc 2023.8) and candidate unpatched (Homebrew glslc 2026.4) completed 32 tokens on both models. The candidate's Qwen output repeated newlines with Homebrew glslc. Rebuilding the candidate with release glslc 2023.8 yielded **32/32 Qwen tokens identical to v1.0.0-4** (14/32 matching CPU) and **32/32 stories15M tokens identical to CPU**. OpenCL ran without an Adreno kernel library reference. On Android 12+, the app needs `<uses-native-library android:name="libOpenCL.so" android:required="false"/>` for OpenCL to load. |
| 2 Pixel 9 Pro, A53 | Firebase Test Lab `caiman`/API35 (Mali-G715), SC-53C/API36 (Mali-G68) | Pixel 9 Pro: Vulkan tokens equal CPU tokens on both models. A53: Vulkan 1.1 is refused by the harness, CPU passes; 0002 reachability is still not shown. |
| 3 ARM and x64 throughput | GCE t2a (Neoverse-N1), c4a (Neoverse-V2), c3 (Xeon 8481C); Qwen2.5-0.5B F16, pp512/tg128, FA on/off, auto and armv8.2 layouts, 6 samples per cell | Candidate against v0.9.0: t2a 0.992 to 1.027, c4a 0.953 to 1.032 (lowest: auto-layout tg128 FA off), x64 1.013 to 1.150. Against v1.0.0-4: 0.971 to 1.031. |
| 4 CUDA | GCE n1-standard-8, Tesla V100 (CC 7.0), driver 580.178.04 | CUDA 12.8.93 with `cuda_nvml_dev` compiles (`nvml.h` SHA-256 `2469cfaa…`, sm_70 only to fit the run cap) and loads; `NVML present`, 25 of 25 layers offloaded, flash attention on and off give identical tokens, and with `libnvidia-ml.so.1` absent the module loads, logs `NVML absent` and produces identical tokens. CUDA and CPU match for 23 of 32 tokens. L4 and T4 were in stockout for 1.5 hours; SKU gating on Ada or Hopper and HIP did not run. |
| 5 Metal | Apple M4 Max, macOS 27.2.1 | CTest 101/101; Metal against CPU: EmbeddingGemma 2 BF16 cosine 0.999993, greedy tokens equal for 10 (Qwen3.5-0.8B), 31 (Qwen3-0.6B) and 15 (SmolLM2-360M) of 32. iPhone, macOS 13/14 and iOS 16/17 did not run: no device harness for the native libraries exists. |
| 6 Windows consumer | none | Did not run. |
| 7 Speculative and TTS | Apple M4 Max (Metal for DSpark, CPU for TTS) | `draft-dspark` with the real Qwen3.5-0.8B DSpark draft and Qwen3.5-0.8B Q4_K_M target: output equals greedy (48/48), 238 drafted, 14 accepted. Pocket TTS english at `n_batch` 256 and 2048, Qwen3-TTS, LFM3-Audio, Soprano and Chatterbox each produce 6 frames. MTP, DFlash, DFlash2 and LiliCorr pairs, ASR stream, ON_DEVICE state, vision and the SVE `Q5_0`/`Q5_1` kernels did not run. |

### Android Vulkan shader features

Android release builds use Ubuntu 24.04's glslc `2023.8-1build1`. It supports
`GL_KHR_cooperative_matrix` and leaves `GL_EXT_integer_dot_product` disabled.
Homebrew glslc 2026.4 compiles the integer-dot q8_1 shaders, which produced
incorrect Qwen Q4_0 output on the S24's Qualcomm proprietary driver. Disabling
integer dot in that build restored the same 32 tokens as the release-glslc
build (Firebase Test Lab `matrix-3743br4xs6xh1` and `matrix-1gj6wnhk5romg`).
Keep the build and artifact guards until the fix in
[llama.cpp #30821](https://github.com/ggml-org/llama.cpp/pull/30821) is in the
pinned llama.cpp and Adreno Vulkan is requalified with integer dot enabled.
Use the release glslc for local Android qualification. On macOS, point
`ANDROID_VULKAN_GLSLC` at a wrapper that runs glslc in an `ubuntu:24.04`
container with the build paths mounted, as in the qualification evidence's
`adreno-diag/adreno-glslc-wrapper.sh`.

The integer-dot guard covers the toolchain-dependent S24 failure. Longer
Qwen prompts (33 prefill tokens) aborted with
`vk::Device::createComputePipeline: ErrorUnknown` on Adreno 750 with both
release-equivalent builds and on Adreno 740 with v1.1.0. The candidate stack
traces reach `ggml_vk_matmul`. Tested quantized models also produced corrupt
output on Adreno 830 and 840 with both releases. The v1.1.0 Vulkan results were
at least as good as v1.0.0-4 on the paired devices. Refusing Vulkan on affected
devices belongs in the `llamadart` consumer.

## Required gates before readiness and publication

Gates 2 to 4 and parts of 5 and 7 ran on 2026-11-26 ([receipts](#hardware-and-model-receipts)); the rest below did not. Each receipt must name owner head,
upstream, patch/source digests, compiler, ISA and backend; model and device
receipts also name archive/module/model/projector hashes and device/OS/driver.

1. **Hosted PR lanes** at the final head: Windows ARM64 ClangCL, Android
   artifact/disassembly and compiled dispatch ([ISA record](v110_android_isa_qualification.md)),
   Linux variants/glibc/packaging/host shutdown, Apple OS floor
   (`metal_os_floor.yml`), CUDA row compile, all ABI rows.
2. **0001 retirement trigger, Firebase Test Lab** (as in the
   [v0.6.0-2 evidence](v060_2_wrapper_rebuild.md)): SC-51E/API36 Adreno 750 old
   unpatched, old patched and candidate unpatched, Vulkan and OpenCL, plus A53
   and Pixel 9 Pro. OpenCL runs must show no attempt to load the Adreno kernel
   library. A failing candidate unpatched run blocks 0001's removal.
3. **ARM FP16 throughput ([#192](https://github.com/leehack/llamadart-native/issues/192))**:
   GCE t2a (Neoverse-N1) and c4a F16 pp512/tg128, FA on and off, against
   v0.9.0, with 0008 and 0009. Acceptance is the ratios already recorded in
   [native192](native192_arm_fp16_f16_route.md); the new CPU-FA F32 V
   accumulator and `mul_mat_one_chunk` blocking change the baseline, so rerun
   rather than reuse v1.0.0 cells.
4. **CUDA** (GCE project `llamadart-cuda-qa-20260916`, personal account,
   approved): compile with CUDA 12.8.1 including `cuda_nvml_dev`, load, device
   enumeration and output parity under the SKU gating that `ggml-cuda.cu`
   now derives from NVML; check behavior with `libnvidia-ml.so.1` absent.
5. **Metal**: iPhone and macOS runs with the new Metal kernels (small-K
   mat-vec) plus `metal_os_floor.yml`; the OS-floor lanes do not cover macOS 13
   and 14 or iOS 16 and 17.
6. **Windows consumer** qualification of the new bundle and the CUDA sidecar.
7. **Model packs**: text, vision, embedding, speculative (MTP, DFlash, DFlash2,
   LiliCorr, DSpark where a pair exists, n-gram by name), the admitted TTS
   families (Qwen3-TTS, LFM3-Audio, Soprano, Chatterbox, Pocket) with the new
   speed default, ASR stream, CPU-FA and ON_DEVICE state controls
   ([#187](https://github.com/leehack/llamadart-native/issues/187)).
8. **Release**: nonpublishing preparation, publication and consumer
   adoption each need separate approval. No manual workflow/device/GPU/model
   run, release, tag or consumer pin is authorized by the pull request.
