# v0.6.0-2 rebuild

Changes on llama.cpp `v0.6.0`
(`d81235049384534c167caea52b85a694f6103d14`) for the native rebuild tag
`v0.6.0-2`. The exported header is unchanged from `v0.6.0-1`.

## Carried llama.cpp patches

This is the first release built with patches on top of the pinned upstream
source. `patches/llama.cpp/README.md` has the conditions, the file format and
the procedure. In short:

- `third_party/llama.cpp` is not modified. While CMake configures,
  `tools/llama_cpp_patches.py` writes a patched copy of each source that a
  patch edits to `<build>/llama.cpp-patched/`, and
  `cmake/llama_cpp_patches.cmake` makes the upstream target compile the copy.
  `cmake/kleidiai_windows.cmake` and the Android CPU backend score object
  already add sources to upstream targets from this repository's CMake; this
  replaces one.
- A hunk applies only where its context and removed lines occur exactly once.
  Every configuration applies every patch, so a patch that upstream has
  outgrown fails every lane, including the automatic stable release.
- `assets.json` gains `llama_cpp_patches`, the `file`, `sha256` and `tracking`
  issue of each patch, and `[]` for a release without any.
- `tools/validate_android_artifacts.py` fails an Android bundle in which a
  library lacks the text that a patch compiles into it.

A consumer that compares against upstream `llama.cpp` at `llama_cpp_commit`
(`docs/parity_tools.md`) compares against unpatched source. The patches of
this rebuild change only `ggml-vulkan`, each for the condition its section
names.

## Adreno 750 shader compiler crash

Issue: [llamadart-native#79](https://github.com/leehack/llamadart-native/issues/79).
Patch: `0001-vulkan-adreno-750-matvec-shared-memory-reduction.patch`.

On a Galaxy S24 (`SC-51E`, Adreno 750, driver version `2150604839`, build
`c81b28e1a5`) the first prompt decode on Vulkan ends the process: `SIGSEGV` at
`0x2d0` in the driver's shader compiler (`CreateQGLCProgram`), under
`vkCreateComputePipelines`. A crash inside the driver cannot be caught.

Root cause at `d81235049`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
`ggml_vk_load_shaders` sets `use_subgroups = device->subgroup_arithmetic`
(line 2899) and with it builds every dequantize matrix-vector pipeline in the
subgroup or hybrid reduction mode (lines 2927 to 2929), requiring full
subgroups of the device's subgroup size, 64 here (lines 2910 and 2936;
`eRequireFullSubgroupsEXT` and `requiredSubgroupSize` at lines 731 and 742).
The pipelines are compiled on first use, from `ggml_vk_mul_mat_vec_q_f16`
(line 6638). This driver's compiler crashes on those shaders once the batch
has enough columns: 5 for F32, 3 for F16, BF16, Q4_0, Q4_1, Q5_0 and Q5_1, 2
for IQ4_NL. A single-token decode compiles, so the crash comes with the first
prompt. Upstream `master` at `609290be6` (2026-10-09) has the same code.

The patch sets `use_subgroups = false` for that block when the device is
exactly this driver (vendor `0x5143`, `VK_DRIVER_ID_QUALCOMM_PROPRIETARY`,
driver version `2150604839`, name `Adreno (TM) 750`), in Android builds only.
The pipelines then use the shared-memory reduction and no required subgroup
size, which is what ggml already does for a device without subgroup
arithmetic and, for part of the block, for the Imagination driver. It is the
source change of the draft
[ggml-org/llama.cpp#29165](https://github.com/ggml-org/llama.cpp/pull/29165)
(`8640bb336`), without its build option.

Evidence, from the issue: Galaxy S24 `SC-51E`, Android 16, Firebase Test Lab,
a JNI app on the llama.cpp C API, context 1024, batch 1024, micro-batch 512,
flash attention off, every layer on the GPU, greedy decoding, 32 tokens,
compared with the CPU in the same app. llama.cpp was `b11256` (`c85b92c69`),
not `v0.6.0`.

| Build | stories15M F32 | Qwen3.5-0.8B Q4_0 |
| --- | --- | --- |
| `b11256` | crash | crash |
| `b11256` with this change | 32 of 32 tokens equal to CPU | 32 of 32 tokens equal to CPU |

What the patch does not fix, on this driver:

- Matrix-vector products of Q8_0, Q2_K to Q6_K, IQ4_XS and IQ3_S weights give
  wrong results (relative error 0.06 to 0.31) with and without the patch, and
  Q4_K with one column fails pipeline creation with `VK_ERROR_UNKNOWN`. A
  model in one of these types, which includes the common `Q4_K_M`, does not
  work on Vulkan on this driver.
- Batches above 8 columns, flash attention, `mul_mat_vec_id` and F16
  activations were not run on the device.
- Another driver version or another Adreno is not matched and behaves as
  before.

Remove the patch when the pinned upstream has an equivalent change, or when
the S24 run passes without it.

## Checks

```bash
git submodule update --init --recursive
python3 -m unittest discover -s tests -p 'test_*.py'
python3 scripts/verify_release_provenance.py
cmake -S . -B build/v060 -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_METAL=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/v060 --parallel 8
ctest --test-dir build/v060 --output-on-failure
```

`tests/test_llama_cpp_patches.py` covers the patch tool on its own sources and,
with the submodule checked out, applies the carried series to it. The
`android-vulkan-shaders` lane of `validate_wrapper.yml` compiles the patched
`ggml-vulkan.cpp` with the release NDK against the pinned, `v0.5.0` and
`v0.6.0` upstreams and, on the pinned one, validates the bundle. No lane runs
ggml-vulkan on a device.
