# Windows Visual C++ runtime

Windows bundles ship the Visual C++ v14 runtime (and, on x64, the OpenMP
runtime) next to the llama.cpp DLLs, so they load on hosts without the Visual
C++ Redistributable, such as stock Windows Server 2022 (issue #95).

## What ships

`tools/build.py windows` copies every DLL the lane's libraries import, except
libraries Windows or a GPU driver provides (`OS_PROVIDED_DLLS` in
`tools/windows_runtime.py`). The core (Vulkan) lane stages all of them, so the
merged bundles gain:

| Bundle | Added | Removed |
| --- | --- | --- |
| `windows-x64` | `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll` | - |
| `windows-arm64` | `msvcp140.dll`, `vcruntime140.dll` | `psapi.dll` (a Windows system file that earlier releases copied from the build host), `libomp140.aarch64.dll` (see below) |

`vcomp140.dll` keeps shipping on x64. Any other `msvcp140_*`,
`vcruntime140_*` or `concrt140` DLL ships only when a bundled library imports
it.

## Source and licence

Runtime and OpenMP DLLs come only from the redistributable folder of the
toolset that built the lane: `%VCToolsRedistDir%\<arch>\Microsoft.VC*.CRT` and
`Microsoft.VC*.OpenMP`. The build fails rather than fall back to
`debug_nonredist`, `System32`, `onecore` or `PATH`, and it never takes any
dependency from the Windows directory. Those folders hold the files the Visual
Studio licence lists as Distributable Code (`redist.txt`), and Microsoft
documents copying them next to the application as app-local deployment.
App-local copies are not serviced by Windows Update; each native release
refreshes them from its build image.

## No OpenMP on arm64

The arm64 lanes build with ClangCL, whose `/openmp` imports the LLVM OpenMP
runtime `libomp140.aarch64.dll`. Visual Studio (checked through 14.51) ships
it only under `debug_nonredist\arm64\Microsoft.VC*.OpenMP.LLVM`, and its
[redistribution terms](https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution)
forbid distributing that folder. Releases through `v0.7.0-1` bundled it
anyway ([#120](https://github.com/leehack/llamadart-native/issues/120)).

The `windows-arm64-full` preset therefore sets `GGML_OPENMP=OFF`, as the
Android and Apple builds already do. ggml's CPU backend then runs graphs on
its own thread pool instead of OpenMP. x64 keeps OpenMP through
`vcomp140.dll`, which is Distributable Code.

The build never resolves `libomp140*`, debug runtimes (`*140d.dll`,
`ucrtbased.dll` and the like) or anything else
`windows_runtime.NON_REDISTRIBUTABLE_PATTERNS` names, and the bundle check
rejects any of them in a bundle.

## Why not the static CRT (`/MT`)

- The OpenMP runtimes and the vcpkg `openblas.dll` import `vcruntime140.dll`,
  so a static CRT in our libraries does not remove the dependency.
- llama.cpp passes CRT objects between its DLLs: `ggml_fopen` in
  `ggml-base.dll` returns a `FILE *` that `llama.dll` reads and closes. With a
  static CRT each DLL has its own stdio state, and that breaks.
- App-local DLLs change packaging only; exports and ABI stay as they are.

## Version skew

The bundled runtime must be at least as new as the MSVC toolset of every
bundled DLL. Lanes run on different images (the CUDA lane stays on
`windows-2022` for its host compiler), and only the core lane ships the
runtime, so the merged bundle is checked against every DLL's MSVC linker
version. ClangCL (arm64) links with lld-link, which records 14.0; both arm64
lanes use the same runner image.

Windows binds a DLL import to a module of that name already loaded in the
process, and otherwise searches the loading DLL's directory
(`LOAD_WITH_ALTERED_SEARCH_PATH`) or the application directory before
`System32`. A fresh process therefore uses the bundled copy even where a
machine-wide runtime exists. A host process that loaded an older runtime first
(for example an application that ships its own `msvcp140.dll`) keeps it, so
an application that ships the runtime itself needs one at least as new as the
version the bundle check prints.

## Checks

`tools/validate_windows_bundle.py` reads PE files on any host. It requires
every import (including delay-loads) to be a bundled DLL or one Windows
provides, every named import from a bundled DLL to be exported there, the
bundled runtime DLLs to share one version, that version to cover every MSVC
linker in the bundle, and no Windows system DLL or `debug_nonredist`-only DLL
to ship. With `--dumpbin` it also compares each import list with
`dumpbin /dependents`.

- `Native Build & Release`: each Windows lane checks its whole output, and
  packaging checks each merged `windows-*` bundle before it is archived.
- `Validate Wrapper`: the x64 core lane (`windows-x64-runtime-closure`) and the
  arm64 Vulkan lane build with `tools/build.py`, run the check, and load
  `llamadart.dll` and `ggml-cpu.dll` from a `/MT` host
  (`tools/windows_load_smoke.c`). The runners have the runtime installed
  machine-wide, so the smoke fails unless every runtime module in the process
  came from the bundle.

The final acceptance in #95, a `validation-model-core` run on stock Windows
Server 2022 without the redistributable, runs from `llamadart` after it adopts
a release with this change.
