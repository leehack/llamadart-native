# Carried llama.cpp patches

Changes to llama.cpp sources that this repository's builds apply on top of the
pinned upstream commit. `third_party/llama.cpp` itself stays exactly as
upstream published it.

Updating the submodule to an upstream release that has the fix is still the
preferred way to get a fix (`AGENTS.md`). A patch is carried only when all of
these hold:

- A released artifact crashes or gives wrong results, and nothing in `src/` or
  the build configuration can prevent it.
- No upstream release has the fix, and the fix is a few lines.
- The patch has an issue in `leehack/llamadart-native` or `leehack/llamadart`
  and, where one exists, an upstream pull request or issue.

## Files

- `series.json` lists the patches in the order they apply. Each entry has the
  patch `file`, the `tracking` issue, the `upstream` pull request or issue to
  watch (a permalink to the upstream lines when there is none), and
  `android_markers`: a `library` of the Android bundle and a `text` that only
  the patched source compiles into it.
- `NNNN-short-name.patch` is a unified diff (`git diff -U2` inside
  `third_party/llama.cpp`) of one existing `.c`, `.cpp`, `.m` or `.mm` file. A
  patch cannot add a file or edit a header: only a source that an upstream
  target compiles can be replaced.

## How they are applied

`cmake/llama_cpp_patches.cmake` runs `tools/llama_cpp_patches.py` while CMake
configures. It writes the patched copy of each source to
`<build>/llama.cpp-patched/` and makes the upstream target compile that copy
instead of the original.

A hunk applies only where its context and removed lines occur exactly once in
the upstream file. There is no fuzz. A patch that does not fit the llama.cpp
being built is skipped whole with a CMake warning that names it, and the build
goes on: `validate_wrapper.yml` builds older upstream commits for
qualification, and those must keep building.

## What keeps a release from losing a patch

`tools/validate_android_artifacts.py` fails an Android bundle in which a
library lacks the `text` of an `android_markers` entry. It reads the built
library, so it does not matter why the patch is missing. The release runs it
on every Android bundle and `validate_wrapper.yml` on the pinned upstream, so
a patch that upstream has outgrown fails the pull request that moves the
upstream, or the release, the automatic stable release included. Check then
whether upstream fixed the defect, and refresh the patch or delete it.

For that reason every patch must have at least one marker, and so must change
a source that an Android library is built from. A patch fits the same
upstream source on every platform or on none, so a release whose Android
bundle has every marker has every patch on its other platforms too.

`python3 tools/llama_cpp_patches.py apply --strict --upstream
third_party/llama.cpp --output <dir>` fails where the build would skip.

## What a release records

`assets.json` has `llama_cpp_patches`: the `file`, `sha256` and `tracking` of
every patch, an empty list when there is none. `llama_cpp_commit` remains the
upstream commit the patches apply to.

## Adding, refreshing or removing a patch

1. Edit the source inside `third_party/llama.cpp`, write
   `git -C third_party/llama.cpp diff -U2 > patches/llama.cpp/NNNN-name.patch`,
   then restore the submodule with `git -C third_party/llama.cpp checkout -- .`.
2. Add or remove the `series.json` entry.
3. Run `python3 -m unittest tests.test_llama_cpp_patches` with the submodule
   checked out, and build a target that compiles the patched source.
4. Describe the patch, its evidence and what removes it in the document of the
   rebuild that introduces it under `docs/`.

Remove a patch when the pinned upstream contains the fix or the defect no
longer reproduces without it.
