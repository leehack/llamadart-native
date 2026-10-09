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
