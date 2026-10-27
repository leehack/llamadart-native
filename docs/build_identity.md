# Build identity

`llama_dart_build_identity()` returns a string that names the inputs a
libllamadart was built from. `llamadart` compares it before it loads a llama.cpp
state file that another process saved: llama.cpp itself checks only the session
file's magic and version, which do not change when the runtime that wrote the
file does.

## Format

One line of `key=value` fields after a format tag, in this order:

```text
llamadart-native-build/1;owner=<commit>[+dirty.<digest>];llama.cpp=<commit>[+dirty.<digest>];ggml=<tree>;patches=<digest>|none;config=<config>;compiler=<cxx>,<c>;target=<target>;backends=<list>|none;options=<digest>
```

| Field | Value |
| --- | --- |
| `owner` | llamadart-native commit. `+dirty.<digest>` digests the uncommitted changes to tracked files, submodule commits included. |
| `llama.cpp` | Commit checked out in `third_party/llama.cpp`, with the same dirty digest. |
| `ggml` | Git tree of `ggml/` in that commit. |
| `patches` | Digest of the sources that the carried patches which fit this llama.cpp produce (`patches/llama.cpp`), or `none` when none applied. |
| `config` | CMake configuration, such as `Release`. |
| `compiler` | C++ and C compiler ID and version. |
| `target` | System, processor and, where CMake has them, the Apple architectures and SDK platform, the Android ABI, the compiler target and architecture. |
| `backends` | ggml backend libraries built, such as `cpu,metal`. |
| `options` | Digest of every boolean `GGML_*` CMake switch and `LLAMADART_CPU_REQUIRED_FEATURES`. |

Digests are the first 16 hex digits of a SHA-256. Values use only
`[A-Za-z0-9._+,-]`, so no path can appear; `tools/build_identity.py` fails the
build otherwise. The string holds no host name, user or time, and the same
inputs give the same string in any build directory or checkout path. The format
tag changes when a field is added, removed or redefined.

The build stamps the identity on every build, not at configure time, so a
commit or an edit after configuring still changes it.

## What equality means

Two libraries with the same non-empty identity were built from the same
tracked sources, patches, compiler, target, backends and boolean ggml switches
(the inputs named under "Format"). Nothing outside those inputs is compared, so
equality is evidence that their state files are compatible, not proof; see the
list below.

Release builds are checked to carry a complete identity
(`tools/validate_build_identity.py`). A `+dirty` marker is allowed there: a
release whose llama.cpp tag differs from the pinned gitlink changes the owner's
tracked tree.

An empty string means the build could not name its inputs: it was not built
from git checkouts of llamadart-native and llama.cpp, or git did not run. An
empty identity matches nothing, itself included, so `llamadart` refuses a state
file from or for such a build.

## What it does not guarantee

- **Equal output.** Restored state matches a fresh evaluation only within
  numerical tolerance; batching and backend scheduling can differ independently
  of the file (leehack/llamadart#1003).
- **The model or context.** The identity names the runtime. The caller still
  checks the model, its adapters and projector, and the context settings.
- **The device chosen at run time.** It names the backends that were built, not
  the GPU, driver or CPU variant that a process selects among them.
- **Inputs outside git.** Untracked files, the worktrees of submodules other
  than llama.cpp, system headers and SDKs other than their platform name, and
  non-boolean CMake settings are not part of it.
- **A different identity means incompatible.** Any input change produces a new
  identity, including ones that leave the state layout as it was.

## Calling it

The string is a constant of the library and stays valid while it is loaded.
The function takes no lock, has no exception barrier because it cannot throw,
and touches no llama.cpp object or state. It therefore needs no exit teardown
guard: teardown frees tracked objects, and this call uses none
(`doc/llama_cpp_exit_teardown.md` in `llamadart`).
