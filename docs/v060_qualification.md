# llama.cpp v0.6.0 qualification

Candidate upstream is tag `v0.6.0`, exact commit
`d81235049384534c167caea52b85a694f6103d14`. Production remains
`v0.5.0` / `7fe450e19305b828c199d602c23a8337aaa1f03b` until qualification
and explicit publication approval. This preparation does not change release
tags, published assets, or downstream native/Web pins.

## Wrapper compatibility

v0.6.0 replaces the image post-decode callback's by-value `llama_batch` with
a read-only `mtmd_helper_embd_batch` view. The adapter preserves the wrapper
callback signature, embedding and section-major M-RoPE positions, sequence
metadata, disabled logits, user data and callback return status. Mutable
callback storage is scoped to the synchronous call and does not cast away
upstream constness.

Speculative processing now takes `common_batch`. Generic, MTP and n-gram
wrapper entry points convert their legacy batches, preserving text/embedding
data, positions, sequence metadata and output flags. Implicit text positions
continue from the draft context's memory. Invalid inputs and nonzero sequence
IDs fail before upstream processes them; wrapper handles own one sequence.
Embedding batches require a known target width, and omitted M-RoPE positions
are rejected. Older upstream builds retain their original forwarding paths.

CMake probes declarations with the wrapper's C++17 standard, without linking
upstream symbols or executing target code. Probes are refreshed when the source
revision changes in an existing build directory.

## Reproduce local checks

```bash
git -C third_party/llama.cpp fetch --depth=1 origin v0.6.0
git -C third_party/llama.cpp checkout --detach d81235049384534c167caea52b85a694f6103d14
cmake -S . -B build/v060 -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_METAL=ON -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DGGML_CPU_KLEIDIAI=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/v060 --parallel 6
ctest --test-dir build/v060 --output-on-failure
python3 -m unittest discover -s tests -p 'test_*.py'
python3 tools/validate_exports.py --format nm --tool nm build/v060/libllamadart.dylib
python3 tools/validate_grammar_boundary.py build/v060/libllamadart.dylib
```

The default macOS tests use tiny generated GGUF fixtures. A real-vocabulary
GGUF is required by the additional `model-speculative-process` scenario:

```bash
build/v060/llamadart_exit_teardown_test model-speculative-process /path/to/model.gguf
build/v060/llamadart_exit_teardown_test model-mtmd /path/to/vision.gguf /path/to/mmproj.gguf
```

Setting `LLAMADART_EXIT_TEARDOWN_TEST_MODEL` when configuring registers the
real speculative process check in CTest. Existing
`LLAMADART_EXIT_TEARDOWN_TEST_MMPROJ` registers the multimodal scenarios.
The speculative check requires actual draft-context advancement over two calls.
The multimodal check covers the null-callback helper path, successful legacy
callbacks and propagation of a nonzero callback result.

## Local evidence and remaining gates

On macOS arm64 / Apple M4 Max, the v0.6.0 Debug Metal build passed 37 default
CTest cases, 175 Python tests, all 65 required wrapper exports and the public
grammar repetition acceptance/rejection boundary. The compatibility build at
v0.5.0 also passed its 37 default CTest cases. Cached SmolLM2-135M-Instruct
Q2_K passed bounded CPU and Metal text generation and actual draft-context
processing. The 32-token text output matched v0.5.0 on each backend for the
same prompt. Cached SmolVLM-500M-Instruct Q8_0 with its F16 projector passed
image evaluation and both success/error legacy callback paths. These are
pipeline checks, not qualification of newly added model families, MTP weights,
ASR or TTS. M-RoPE layout has synthetic adapter coverage, not a real-model claim.

`validate_wrapper.yml` adds exact v0.6.0 candidate lanes for Android ISA and
Vulkan shaders, emulated KleidiAI dispatch, Windows ARM64 CPU/Vulkan, Linux
x64/arm64 artifacts, Linux wrapper contracts, MSVC mtmd linking and macOS
Release/ASan teardown. Existing production and historical baselines remain.
All 36 wrapper jobs passed at implementation head
`41d697cbc132d9d198ec03fb8a44cdfe4fba7b80` in
[run 37604135940](https://github.com/leehack/llamadart-native/actions/runs/37604135940).
The three release provenance jobs also passed in
[run 37604135946](https://github.com/leehack/llamadart-native/actions/runs/37604135946).

The Android source-pair audit and local production ISA validator passed as
recorded in `v060_android_isa_qualification.md`. The complete source pair is
accepted without broadening the instruction/function allowlist. Hosted ISA/dispatch/platform checks passed at the implementation head above.
Every updated PR head must pass its required checks, and independent review
remains required before mark-ready. Native full-platform bundles, manifest checksums, Apple packaging
and downstream regenerated bindings/model-backed qualification remain separate
artifact/adoption gates. The matching Web bridge must be qualified and published
in its owning repositories before its pin moves.
