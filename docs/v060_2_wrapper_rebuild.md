# v0.6.0-2 wrapper rebuild

Wrapper-only changes on llama.cpp `v0.6.0`
(`d81235049384534c167caea52b85a694f6103d14`) for the native rebuild tag
`v0.6.0-2`. The exported header gains no declaration and loses none: the 85
symbols of `v0.6.0-1` are the 85 of this rebuild. Only Linux builds change;
the macOS library built from this tree is byte-for-byte the one built from
`v0.6.0-1` (see "Checks").

## Exit on Linux

Issue: [llamadart#949](https://github.com/leehack/llamadart/issues/949). A C
`exit()` while another thread is inside llama.cpp killed the process on
Linux. `exit()` destroys C++ statics that the thread still reads:

| Frame of the crash | Static | Created |
| --- | --- | --- |
| `ggml_cpu_extra_compute_forward`, `ggml_cpu_extra_work_size` | the vector in `ggml_backend_cpu_get_extra_buffer_types` (`libggml-cpu.so`) | on first use, before the first decode |
| `ggml_backend_dev_type` under `llama_model_load_from_file` | a device of ggml's registry; not traced to one static | before the load |
| `unicode_utf8_to_byte` under `llama_vocab::impl::load` | the map in `unicode_utf8_to_byte` (`libllama.so`) | in the middle of the first model load |

The third one is why registering an exit handler is not enough. `exit()`
runs one list of handlers and static destructors in reverse order of
registration, so no handler that libllamadart registers before a load runs
before the destructor of a static that the load creates. On Apple platforms
llama.cpp is part of `libllamadart.dylib`, and a definition of `__cxa_atexit`
inside that image orders teardown before all of them. On Linux they are
separate libraries, some of them modules that ggml loads and unloads while it
looks for the best backend.

### What changes

1. The llama.cpp libraries register no static destructors.
   `src/llama_dart_static_destructors.c` defines a hidden `__cxa_atexit` that
   registers nothing, and `CMakeLists.txt` adds it to every shared library and
   backend module under `third_party/llama.cpp`, whatever backends the build
   enables. The statics stay valid until the process is gone: also for a call
   that teardown does not wait for, and for one that outlasts its wait.
   Upstream sources are not patched.
2. C `exit()` runs the wait of exit teardown. libllamadart registers the
   handler at the first creating call (`llama_dart_model_load_from_file`,
   `llama_dart_init_from_model`, `llama_dart_mtmd_init_from_file`, the TTS,
   speculative, MTP and n-gram init functions), so that an exit during the
   first load finds one, and again when the first object is tracked. The wait is the one of
   `llama_dart_exit_teardown`: up to 2 s (`llama_dart_exit_set_wait_ms`) for
   the calls in flight, a model load is cancelled, then 250 ms after the last
   call, and from then on every thread that reaches libllamadart is blocked.
   The exit handlers and destructors of other libraries run after it.
3. The exit frees nothing. Apple frees the tracked objects because
   ggml-metal aborts over a live buffer. Nothing on Linux does, the issue
   found an exit with a model left loaded clean on CPU, Vulkan and CUDA, and a
   free during `exit()` would run among the exit handlers of the GPU driver in
   an order that nothing controls. `llama_dart_exit_teardown` called by a
   native host still frees.
4. While teardown waits, a decode or an encode on the CPU backend ends with
   llama.cpp's status for an aborted evaluation (2).
   `llama_dart_init_from_model` sets the context's abort callback for that
   when the caller passes none. Without it a long decode outlasts the wait and
   is still running when the exit reaches other libraries: with the BLAS
   module, OpenBLAS's destructor then never returns (see "Measurements").
   llama.cpp logs three error lines for the ended decode.
5. libllamadart itself binds `__cxa_atexit` to the definition that Apple
   uses, so a static of its own is destroyed only after the wait.

`tools/validate_linux_artifact.py`, which the release runs on every Linux
archive, fails an archive in which a library built here imports
`__cxa_atexit`.

### Why not a gate in front of every destructor

[`v060_1_wrapper_rebuild.md`](v060_1_wrapper_rebuild.md) recommended a hidden
`__cxa_atexit` in every library that registers each destructor behind a gate
exported by `libggml-base.so`, which libllamadart points at teardown. The wait
without frees is taken from there. The gate is not:

- ggml unloads a backend module that it loaded only to score it, or that
  failed to initialize. `dlclose` runs the destructors that the module
  registered, and each would run the gate, which is teardown, in the middle of
  a run.
- A destructor behind a gate still runs once the wait is over. A call that
  outlasts the wait would crash as before, which was 10 of 10 for a long
  decode with a slow host below.
- It would add an exported symbol to an upstream library.

### What it does not cover

- An exit handler that another library registers after the two handlers above
  runs before the wait, under the calls in flight. A GPU driver that
  registers one late, during a decode for example, is in that position.
  Statics that llama.cpp creates late no longer matter.
- A call that is not a call in flight is not waited for, as on Apple. Its
  statics stay valid on Linux; what it uses of other libraries may not.
- A call in flight that outlasts the wait and cannot be ended: a decode on a
  GPU backend, which does not read the abort callback, an image or audio
  evaluation, or a context with an abort callback of its own. The exit then
  goes on with the call running.
- `quick_exit` and `_exit` run no handler, as before. Neither destroys
  statics.
- A child of `fork` that calls `exit()` runs the handler with the registry it
  copied: it waits out the wait time for calls that are not its own.
- An exit handler that code inside a llama.cpp library registers with
  `atexit` binds to the same definition and is dropped. No backend built here
  registers one; a library such as the CUDA runtime, which the modules load
  as a library of its own, is not affected.

### Other platforms

- musl: nothing here is specific to glibc. The definition binds at link
  time, and libllamadart reaches the C library's `__cxa_atexit` through
  `dlsym(RTLD_NEXT)`. No bundle is built for musl and none was run.
- Android: left as it was, by `!defined(__ANDROID__)` in the two sources and
  `CMAKE_SYSTEM_NAME STREQUAL "Linux"` in CMake. bionic orders exit handlers
  the same way, but the issue has no Android measurement and nothing here ran
  on Android.
- Windows: not applicable, going by the documented behavior of the C runtime
  and not by a run. `exit()` ends in `ExitProcess`, which terminates
  the other threads before any DLL destroys its statics, so a thread is gone
  before the statics it read are destroyed. A handler in `llamadart.dll` runs
  at `DLL_PROCESS_DETACH`, under the loader lock and after those threads were
  terminated: it could wait for nothing, and a lock that a terminated thread
  held stays held. A wait there would have to run in the host before
  `ExitProcess`, which is `llama_dart_exit_teardown`'s job. No Windows crash
  of this kind is known, and no lane could show one.

## Measurements

Linux arm64 container (Ubuntu 24.04, glibc 2.39, GCC 13.3) on an Apple M4
Max, the `linux-arm64-full` preset without KleidiAI: Release with link-time
optimization, OpenMP, the CPU, Vulkan and BLAS modules. A probe opens
`libllamadart.so` with `dlopen`, as Dart does, and calls C `exit()` from the
main thread while another thread is inside a guarded call. "Slow host" adds
an exit handler of the probe's own that takes 300 ms, registered before the
library is opened and so run last, as the handlers of a larger host are.
Vulkan is Mesa 25.2.8 lavapipe (`llvmpipe`, `GGML_VK_VISIBLE_DEVICES=0`, all
layers offloaded); BLAS is OpenBLAS 0.3.26. The numbers are failed runs of
10: a signal, an exit code other than 0, a failed `GGML_ASSERT`, or no exit
within 120 s.

CPU and BLAS rows use Qwen3.5 0.8B Q4_0, Vulkan rows `stories15M`.

| Exit | Modules | Host | `v0.6.0-1` | This rebuild |
| --- | --- | --- | --- | --- |
| while generating (decode and sample in a loop) | CPU | fast | 7 | 0 |
| | CPU | slow | 10 | 0 |
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 5 | 0 |
| | Vulkan | slow | 10 | 0 |
| inside one decode of 1500 tokens | CPU | fast | 1 | 0 |
| | CPU | slow | 10 | 0 |
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 0 | 0 |
| | Vulkan | slow | 6 | 0 |
| 100 ms into the first load | CPU | slow | 9 | 0 |
| 400 ms into the first load | CPU | slow | 6 | 0 |
| 200 ms and 800 ms into the first load | CPU | fast | 0, 0 | 0, 0 |
| 5 ms and 60 ms into the first load | Vulkan | fast | 0, 0 | 0, 0 |
| | Vulkan | slow | 10, 10 | 0, 0 |
| first load, once tensors are loading | CPU, Vulkan | fast, slow | 0 | 0 |
| halfway through a second load | CPU | slow | 1 | 0 |
| | CPU, Vulkan | fast | 0 | 0 |
| `return` from `main` while generating | CPU | fast | 8 | 0 |
| | Vulkan | fast | 3 | 0 |
| model idle; everything freed; `return` from `main` with a model idle | CPU, BLAS, Vulkan | fast, slow | 0 | 0 |

`v0.6.0-1` is its source, `ceae6f7`, built in the same container. Every
failed run of it ended in a signal: `SIGSEGV`, and in five lavapipe runs
`SIGABRT` (`double free or corruption`). The scenarios that were clean before
are clean after, with the model left loaded and not freed.

Two measurements on the way decided parts of the design:

- Without the abort callback, the 1500-token decode with the BLAS module
  hung in OpenBLAS's destructor (`gotoblas_quit`, `blas_thread_shutdown_`) in
  6 of 10 exits and was killed after 120 s; with the callback, 0 of 10.
  Without the BLAS module the same exit was clean 10 of 10 either way: the
  decode outlasted the wait and ran on, with its statics intact, until the
  process was gone.
- A build that also freed the tracked objects during `exit()` was clean in
  the 16 lavapipe scenarios it ran (10 of 10 each). That shows nothing about
  the NVIDIA driver, which is why the frees stay out.

On lavapipe the abort callback does not end the 1500-token decode: ggml-vulkan
does not read it.

## Checks

```bash
cmake -S . -B build/wrapper-contract -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DLLAMADART_BUILD_TESTS=ON
cmake --build build/wrapper-contract --target llamadart_exit_teardown_test
ctest --test-dir build/wrapper-contract -R exit_teardown --output-on-failure
for library in build/wrapper-contract/libllamadart.so build/wrapper-contract/bin/*.so; do
  python3 tools/validate_exports.py --format readelf --tool readelf \
    --imports-only --forbid-import __cxa_atexit "$library"
done
```

`llamadart_exit_teardown_test` now runs its model scenarios on Linux as well,
on the CPU backend module and the models it writes itself. In the container
above:

- The release preset: 48 of 48 CTest cases, and the 42 exit cases ten times
  over. The same 42 four times in a RelWithDebInfo build with
  AddressSanitizer, with no report.
- The configuration of the `wrapper-contract` lane (Debug, no OpenMP) on the
  pinned upstream and on the post-v0.4.0 one: 48 of 48 each, and the 42 exit
  cases five times over.
- The same test source against the libraries of `v0.6.0-1`: `model-decode`,
  `model-late-load` and `model-decode-outlasts-wait` end in `SIGSEGV`, and
  `model-load` and `model-free-in-flight` report that the exit did not wait,
  5 of 5 each.
- Each part removed in turn fails one scenario: the abort callback
  `model-decode-ended`, the handler at the first creating call `model-load`,
  the handler at the first tracked object `exit-in-flight`, the exit that
  frees nothing `exit`, and the static destructor object
  `model-decode-outlasts-wait` (`SIGSEGV`). Without that object
  `validate_linux_artifact.py` names seven libraries of the archive.

`validate_exports.py --forbid-import` now reads ELF imports with
`--format readelf`. With `--format nm` it reads names as Mach-O ones, without
their leading underscore, and an ELF `__cxa_atexit` never matched.

macOS arm64 (Apple M4 Max), `macos-arm64-full` preset: `libllamadart.dylib`
built from this tree and from `v0.6.0-1` at the same path have the same
SHA-256, and 46 of 46 CTest cases pass (45 before; `exit-in-flight` is new).
186 Python tests pass (183 before), and the 20 of the Linux artifact tools
(18 before).

## Not verified

- A real Vulkan GPU and CUDA. The container has neither; lavapipe is a CPU
  implementation with its own threads and exit handlers, which is a different
  driver from the NVIDIA one of the issue. The run that settles it is the one
  of the issue on its host (GCE L4 VM, Ubuntu 24.04, NVIDIA 580): the
  `quit-loaded`, `quit-generating`, `quit-loading`, `return-loaded` and
  dispose-then-quit scenarios of llamadart's
  `test/fixtures/llama_cpp_exit_probe.dart` on the `v0.6.0-2` bundle, with the
  CPU, Vulkan and CUDA backends, at least 10 runs each.
- Linux x64. The measurements are arm64; `validate_wrapper.yml` runs the
  scenarios on x64.
- HIP and musl builds, Android and Windows.
- A Dart or Flutter process. The probe is C++; llamadart's own probe runs
  against a published bundle.
