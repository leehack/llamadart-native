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

1. No library of the bundle registers static destructors.
   `src/llama_dart_static_destructors.c` defines a hidden `__cxa_atexit` that
   registers nothing, and `CMakeLists.txt` adds it to libllamadart and to
   every shared library and backend module under `third_party/llama.cpp`,
   whatever backends the build enables. The statics stay valid until the
   process is gone, also for a call that libllamadart does not count.
   Upstream sources are not patched.
2. libllamadart registers an exit handler with `on_exit`, which gets the exit
   status: at the first creating call (`llama_dart_model_load_from_file`,
   `llama_dart_init_from_model`, `llama_dart_mtmd_init_from_file`, the TTS,
   speculative, MTP and n-gram init functions), so that an exit during the
   first load finds one, and again when the first object is tracked.
3. With a call in flight on another thread, the handler ends the process:
   `fflush(NULL)` and `_exit(status)`. That is what `exit()` from `dart:io`
   does on Linux. The handlers that would have run next are those of other
   libraries, a GPU driver or OpenBLAS among them, and they would run under
   the call. It does not wait for the call first. An earlier version of this
   change did, and time inside `exit()` turned out not to be harmless: see
   "Why the exit does not wait".
4. With no call in flight the handler returns at once and the exit goes on.
   It frees nothing. Apple frees the tracked objects because ggml-metal
   aborts over a live buffer. Nothing on Linux does, the issue found an exit
   with a model left loaded clean on CPU, Vulkan and CUDA, and a free during
   `exit()` would run among the exit handlers of the GPU driver in an order
   that nothing controls. `llama_dart_exit_teardown` called by a native host
   still waits and frees, as on every platform.
5. The exit blocks no thread. Apple blocks every thread that reaches
   libllamadart after teardown, because the objects it holds are gone. On
   Linux they are not, and a blocked thread hangs a host whose own exit
   handler joins it. Once the exit has passed libllamadart, on a thread other
   than the exiting one:
   - `llama_dart_exit_free` and the session free functions free nothing, and
     `llama_dart_exit_track` and `llama_dart_exit_untrack` return `false`;
   - a call in flight that begins returns the failure value of its function,
     with `the process is exiting` as `llama_dart_last_error`, and starts
     nothing;
   - `llama_dart_exit_call_begin` counts nothing.
6. `libllamadart.so` is linked with `-z nodelete`: `on_exit` handlers are not
   removed by `dlclose`, so the library must stay loaded.

`tools/validate_linux_artifact.py`, which the release runs on every Linux
archive, fails an archive in which a library built here imports
`__cxa_atexit`, or in which `libllamadart.so` can be unloaded.

### Why the exit does not wait

The first two versions of this change waited at exit for the call in flight,
as Apple does, and then let the exit go on. Two rows of the measurements got
worse than `v0.6.0-1` that way:

- A Dart VM aborts when one of its isolates collects garbage while C `exit()`
  is under way (`runtime/vm/handles_impl.h: 39: error: unreachable code`,
  [llamadart#977](https://github.com/leehack/llamadart/issues/977)). A Dart
  program with an isolate that decodes UTF-8 in a loop and an exit handler
  that sleeps 300 ms, with no llama.cpp in the process, aborts 10 of 10. An
  exit inside a decode of 1500 tokens with such an isolate aborted 17 of 30
  when the exit waited for the decode to end, against 10 of 30 before the
  change. Waiting 20 ms and then ending the process: 17 of 30 as well. Ending
  the process at once: 0 of 30.
- A decode on a GPU backend cannot be ended, as ggml-vulkan does not read the
  abort callback. On lavapipe a decode of 1500 tokens outlasted the two
  seconds, and the exit that then went on under it failed 2 of 50 (6 of 50 in
  the audit) where it had been clean before, with the driver's own exit
  handlers meeting a decode that was still running.

Not waiting at all, and letting the exit go on at once, does not do either:
in the audit that brought the Dart row down to 2 of 30, but OpenBLAS's
destructor then hung under a decode 4 of 10, and the lavapipe decode with a
slow host failed 26 of 50, as it did before the change.

Both are the exit going on, or taking its time, under a call in flight, and
ending the process is what removes that. So the handler does it for every
call in flight, also for one on the CPU backend that would have ended in a
millisecond: that keeps what an exit does from depending on how long a decode
happens to take. The cost is the same in every case and is listed under "What
it does not cover".

### Why not a gate in front of every destructor

[`v060_1_wrapper_rebuild.md`](v060_1_wrapper_rebuild.md) recommended a hidden
`__cxa_atexit` in every library that registers each destructor behind a gate
exported by `libggml-base.so`, which libllamadart points at teardown. Not
freeing at exit is taken from there. The gate is not:

- ggml unloads a backend module that it loaded only to score it, or that
  failed to initialize. `dlclose` runs the destructors that the module
  registered, and each would run the gate in the middle of a run.
- A gate waits, which is what "Why the exit does not wait" rules out.
- It would add an exported symbol to an upstream library.

### What it does not cover

- An exit with a call in flight skips what the exit would have run after
  libllamadart's handler: the exit handlers that were registered before it,
  the host's among them, the static destructors of the host and of other
  libraries, and the flush of anything but the standard C streams. A host
  that needs those has to stop its calls before it exits. `std::cout` is
  flushed only as far as it is synchronized with `stdout`.
- An exit handler that another library registers after the two of
  libllamadart runs before them, under the calls in flight. A GPU driver that
  registers one late, during a decode for example, is in that position, and
  so is a slow one in a Dart process.
- A call that is not a call in flight is not counted, as on Apple: the exit
  goes on under it. Its statics stay valid on Linux; what it uses of other
  libraries may not.
- An exit that falls between two guarded calls of a worker finds none in
  flight and goes on. The worker's next call is then refused: a native host
  gets the failure value (`INT32_MIN`, -1, `NULL` or `false`) with `the
  process is exiting`, where its thread used to crash or, in the first
  versions of this change, to block.
- `llama_dart_exit_teardown` still blocks the threads that reach libllamadart
  after it, on Linux as elsewhere, because it frees what they hold. A host
  that calls it and then joins such a thread in an exit handler hangs, as
  before this change.
- A Dart process in which an isolate runs Dart code during an exit that goes
  on, that is one with no call in flight, still aborts in the Dart VM at the
  rate it did before the change (see "Dart"). The handler of libllamadart
  adds no time to such an exit.
- `quick_exit` and `_exit` run no handler, as before. Neither destroys
  statics.
- A child of `fork` that calls `exit()` runs the handler with the registry it
  copied: with a call in flight in the parent at the time of the fork, the
  child ends as with `_exit`.
- `dlclose` no longer unloads `libllamadart.so` or the libraries it depends
  on.
- An exit handler that code inside a library of the bundle registers with
  `atexit` binds to the same definition as the static destructors and is
  dropped. No backend built here registers one; a library such as the CUDA
  runtime, which the modules load as a library of its own, is not affected.
- The standard streams of a host built against libstdc++ older than GCC 13.
  There `<iostream>` gives every translation unit a static `ios_base::Init`,
  and `std::cout` is flushed when the last of them is destroyed. One inside a
  library of the bundle is now never destroyed, so a host that turned off
  `sync_with_stdio` loses what `std::cout` still buffers at exit (seen in the
  audit with GCC 11.4 and a test library). The release builds its Linux
  libraries with GCC 13, where the stream initializer lives in libstdc++,
  except `libggml-hip.so` (the ROCm image has GCC 11.4), and neither the
  sources of that module nor `hip/hip_runtime.h` include `<iostream>`; the
  other ROCm headers were not checked. A bundle built with an older GCC is
  affected.
- The heap that the statics of a backend module own is not freed when ggml
  unloads the module. The audit measured 544 KB after 300 calls of
  `ggml_backend_load_all_from_path` on a host without a Vulkan driver, where
  the Vulkan module is loaded and unloaded each time, against 36 KB before:
  about 1.7 KB for each unload. llamadart loads the backends once.

### Other platforms

- musl: not supported and not run. No bundle is built for it. `on_exit` is a
  glibc function, so the handler is compiled only where `__GLIBC__` is
  defined; a musl build would keep its statics and do nothing at exit.
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
optimization, OpenMP, the CPU, Vulkan and BLAS modules. `v0.6.0-1` is its
source, `ceae6f7`, built in the same container.

### Native host

A probe opens `libllamadart.so` with `dlopen`, as Dart does, and calls C
`exit()` from the main thread while another thread is inside a guarded call.
"Slow host" adds an exit handler of the probe's own that takes 300 ms,
registered before the library is opened and so run last, as the handlers of a
larger host are. Vulkan is Mesa 25.2.8 lavapipe (`llvmpipe`,
`GGML_VK_VISIBLE_DEVICES=0`, all layers offloaded); BLAS is OpenBLAS 0.3.26.
CPU and BLAS rows use Qwen3.5 0.8B Q4_0, Vulkan rows `stories15M`. The numbers
are failed runs of 10, or of 50 where it says so: a signal, an exit code other
than 0, a failed `GGML_ASSERT`, or no exit within 120 s.

| Exit | Modules | Host | `v0.6.0-1` | This rebuild |
| --- | --- | --- | --- | --- |
| while generating (decode and sample in a loop) | CPU | fast | 7 | 0 |
| | CPU | slow | 10 | 0 |
| | CPU, BLAS | fast | 10 | 0 |
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 4 | 0 |
| | Vulkan | slow | 10 | 0 |
| inside one decode of 1500 tokens | CPU | fast | 1 | 0 |
| | CPU | slow | 10 | 0 |
| | CPU, BLAS | fast | 6 | 0 |
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 0 of 50 | 0 of 50 |
| | Vulkan | slow | 16 of 50 | 0 of 50 |
| 100 ms and 400 ms into the first load | CPU | slow | 8, 8 | 0, 0 |
| 200 ms and 800 ms into the first load | CPU | fast | 0, 0 | 0, 0 |
| 5 ms and 60 ms into the first load | Vulkan | fast | 0, 0 | 0, 0 |
| | Vulkan | slow | 10, 9 | 0, 0 |
| first load, once tensors are loading | CPU, Vulkan | fast, slow | 0 | 0 |
| halfway through a second load | CPU | slow | 5 | 0 |
| | CPU, Vulkan | fast | 0 | 0 |
| `return` from `main` while generating | CPU | fast | 5 | 0 |
| | CPU | slow | 10 | 0 |
| | Vulkan | fast | 7 | 0 |
| | Vulkan | slow | 10 | 0 |
| model idle; everything freed; `return` from `main` with a model idle | CPU, BLAS, Vulkan | fast, slow | 0 | 0 |
| a host handler that joins an idle worker, which frees its model | CPU | fast | 0 | 0 |
| a host handler that joins a worker that is generating | CPU | fast | 10 | 0 |

Every failed run of `v0.6.0-1` ended in a signal: `SIGSEGV`, and on lavapipe
also `SIGABRT` (`double free or corruption`). No row of this rebuild has a
failed run, and none is worse than before. In the last row the host's handler
does not run: the process ends at libllamadart's handler, with the exit
status. In the row above it the handler runs and joins the worker, before and
after.

Other shapes of host, from the probe of the audit, CPU with a fast host
unless it says otherwise, failed runs of 10:

| Exit | `v0.6.0-1` | This rebuild |
| --- | --- | --- |
| from a worker thread while the main thread generates | 9 | 0 |
| from inside an evaluation callback; from inside a load progress callback | 10, 10 | 0, 0 |
| while two models generate | 10 | 0 |
| while a thread generates with the upstream `llama_decode` on a tracked context | 5 | 0 |
| while a thread generates on a model and context from the upstream functions | 8 | 0 |
| while a thread tokenizes in a loop | 2 | 0 |
| inside a long decode on a context with an abort callback of its own, BLAS, fast and slow host | 8, 10 | 0, 0 |
| after `llama_dart_exit_teardown`; after `dlclose` of libllamadart; `quick_exit`; `_exit` | 0 | 0 |
| ten load and free cycles, then right after a generation step | 9 | 3 |

The three failed runs of the last row are not crashes. The exit fell between
two guarded calls of the worker, so it went on, the worker's next decode was
refused (`INT32_MIN`, `the process is exiting`), and that probe ends the
process with a code of its own when a decode fails. A host whose worker
treats a failed call as fatal sees that during an exit; before, the same row
crashed 9 of 10.

The two rows that earlier versions of this change made worse are the long
decode on lavapipe with a fast host (0 of 50 before, 2 of 50 when the exit
waited two seconds and went on, 0 of 50 now) and the Dart row of "Why the
exit does not wait".

### Dart

Dart 3.13.1, CPU. `tests/manual/dart_exit_probe.dart` calls a guarded decode
and sample per step through a small C shim, on `stories15M`; llamadart rows
are its own `test/fixtures/llama_cpp_exit_probe.dart` and a variant with an
isolate that wakes on a 5 ms timer, on llamadart `9275de8` with the bundle
under test and Qwen3.5 0.8B Q4_0. The main isolate calls C `exit()` through
FFI. "Looping" is a second isolate that decodes UTF-8 in a loop. Failed runs
of 30 unless it says otherwise.

| Exit | Other isolate | `v0.6.0-1` | This rebuild |
| --- | --- | --- | --- |
| llamadart `quit-generating` | none | 18 | 0 |
| llamadart `quit-loaded`, `quit-loading`, `return-loaded`, `kill-loading` | none | 0 | 0 |
| llamadart, right after a generation finished | 5 ms timer | 0 | 0 |
| llamadart, 1 s after a generation finished | 5 ms timer | 0 | 0 |
| llamadart, while generating | 5 ms timer | 25 | 0 |
| llamadart `throw-generating`, `throw-loading` (exit code 255 expected) | none | 255, 10 of 10 | 255, 10 of 10 |
| worker isolate generating | none | 15 | 0 |
| worker isolate generating | looping | 12 | 0 |
| worker isolate generating, slow host | looping | 30 | 1 |
| worker isolate inside one decode of 1500 tokens | looping | 15 of 50 | 0 of 50 |
| worker isolate inside one decode of 1500 tokens, slow host | looping | 50 of 50 | 0 of 50 |
| worker isolate idle after a generation | none | 0 | 0 |
| worker isolate idle after a generation | 5 ms timer | 0 | 0 |
| worker isolate idle after a generation | looping | 12 of 100 | 11 of 100 |
| right after a guarded call returned | looping | 36 of 100 | 24 of 100 |
| model idle for 400 ms | looping | 23 of 100 | 20 of 100 |
| `dart:io` `exit()` while generating; `main` returns while generating | none | 0 | 0 |
| worker generating; a 300 ms exit handler registered right before the exit | none | 25 | 19 |
| worker inside the long decode; the same late handler | looping | 30 | 30 |

The failures of `v0.6.0-1` with a call in flight are the `SIGSEGV` of the
issue, or the abort of the Dart VM. All the failures of this rebuild are the
abort of the Dart VM, in exits that go on:

- The three rows with a looping isolate and no call in flight. The VM aborts
  there whether libllamadart has a handler or not, and at a rate that moves
  with the load of the machine: the 100 runs of a row were taken in turns of
  ten for each build, an earlier sample of 30 gave 0 to 4 for `v0.6.0-1` and 0
  to 1 for this rebuild, and the audit saw 1 of 90 and 0 of 90. The handler
  returns at once in these exits and adds nothing to them.
- The one of 30 with a slow host: the exit fell between two guarded calls of
  the worker, found none in flight, and went on into the 300 ms handler.
- The last two rows, where a handler that was registered after libllamadart's
  runs before it and takes its 300 ms first. libllamadart cannot run before a
  handler that is registered later.

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
on the CPU backend module and the models it writes itself. CTest finds the
llama.cpp libraries of a Linux preset build, which keeps them in `bin/`,
without `LD_LIBRARY_PATH`. In the container above:

- The release preset: 52 of 52 CTest cases, ten times over.
- The configuration of the `wrapper-contract` lane (Debug, no OpenMP) on the
  pinned upstream and on the post-v0.4.0 one: 52 of 52 each, and the 46 exit
  cases five times over.
- A RelWithDebInfo build with AddressSanitizer: the 46 exit cases three times
  with leak detection off, with no report. With it on, `barrier-grammar`
  reports 80 bytes that llama.cpp leaks when a lazy grammar is rejected
  (`llama-sampler.cpp:2777`), as before this change, and nothing else does.
- The same test source against the libraries of `v0.6.0-1`, five runs each:
  `model-decode`, `model-unguarded-decode` and `model-exit-join-generating`
  end in `SIGSEGV` every time and `model-late-load` four times;
  `exit-in-flight`, `exit-status`, `model-load` and `model-free-in-flight`
  fail because the exit went on under a call in flight; `exit-late-calls` and
  `model-exit-join-idle` fail because a free after the exit still freed.
- Each part removed in turn fails a scenario:

  | Removed | Fails |
  | --- | --- |
  | the static destructor object | `model-unguarded-decode` (`SIGSEGV`); `validate_linux_artifact.py` names the libraries |
  | ending the process for a call in flight | `exit-in-flight`, `exit-status`, `model-load`, `model-free-in-flight` |
  | the handler at the first creating call | `model-load` |
  | the handler at the first tracked object | `exit-in-flight`, `exit-status`, `exit-late-calls` |
  | the exit status | `exit-status` |
  | the flush of the standard streams | `exit-status` |
  | returning at once with no call in flight (250 ms added) | `model-exit-idle` |
  | threads not blocked after the exit | `exit-late-calls`, `model-exit-join-idle` |
  | the refusal of a late call | `exit-late-calls` |
  | the exemption of the exiting thread | `exit-late-calls` |
  | `-z nodelete` | `validate_linux_artifact.py` |

`validate_exports.py --forbid-import` now reads ELF imports with
`--format readelf`. With `--format nm` it reads names as Mach-O ones, without
their leading underscore, and an ELF `__cxa_atexit` never matched.

No lane runs Dart or a GPU decode. `tests/manual/dart_exit_probe.dart`, with
the shim next to it, is the probe of the "Dart" table; the lavapipe rows need
a build with `GGML_VULKAN=ON`, Mesa's `mesa-vulkan-drivers` and
`GGML_VK_VISIBLE_DEVICES=0`.

macOS arm64 (Apple M4 Max), `macos-arm64-full` preset: `libllamadart.dylib`
built from this tree and from `v0.6.0-1` at the same path have the same
SHA-256, and 46 of 46 CTest cases pass (45 before; `exit-in-flight` is new).
186 Python tests pass (183 before), and the 22 of the Linux artifact tools
(18 before).

## Not verified

- A real Vulkan GPU and CUDA. The container has neither; lavapipe is a CPU
  implementation with its own threads and exit handlers, which is a different
  driver from the NVIDIA one of the issue. The run that settles it is the one
  of the issue on its host (GCE L4 VM, Ubuntu 24.04, NVIDIA 580): the
  `quit-loaded`, `quit-generating`, `quit-loading`, `return-loaded` and
  dispose-then-quit scenarios of llamadart's
  `test/fixtures/llama_cpp_exit_probe.dart` on the `v0.6.0-2` bundle, with the
  CPU, Vulkan and CUDA backends, at least 10 runs each, against `v0.6.0-1`.
  The scenarios with a call in flight end in `_exit` on every backend, which
  the container shows; what a GPU host adds is the exits that go on: a model
  left loaded, and a handler of the driver that was registered after
  libllamadart's.
- Linux x64 by hand. The measurements are arm64; `validate_wrapper.yml` runs
  the scenarios on x64.
- HIP and musl builds, Android and Windows.
- Flutter. The Dart rows are the standalone VM, and whether a Flutter Linux
  window close reaches C `exit()` with a worker inside a guarded call is not
  known. If it does, the exit ends there and the engine's own exit handlers
  do not run.
