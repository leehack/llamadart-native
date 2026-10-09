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
2. C `exit()` runs the wait of exit teardown and nothing else. libllamadart
   registers the handler at the first creating call
   (`llama_dart_model_load_from_file`, `llama_dart_init_from_model`,
   `llama_dart_mtmd_init_from_file`, the TTS, speculative, MTP and n-gram init
   functions), so that an exit during the first load finds one, and again when
   the first object is tracked. It waits up to 2 s
   (`llama_dart_exit_set_wait_ms`) for the calls in flight and cancels a model
   load. With no call in flight it returns at once: it does not let the
   threads settle for 250 ms as `llama_dart_exit_teardown` does, because it
   frees nothing, and because time spent inside `exit()` is not harmless (see
   "Dart").
3. The exit frees nothing. Apple frees the tracked objects because
   ggml-metal aborts over a live buffer. Nothing on Linux does, the issue
   found an exit with a model left loaded clean on CPU, Vulkan and CUDA, and a
   free during `exit()` would run among the exit handlers of the GPU driver in
   an order that nothing controls. `llama_dart_exit_teardown` called by a
   native host still frees.
4. The exit blocks no thread. Apple blocks every thread that reaches
   libllamadart after teardown, because the objects it holds are gone. On
   Linux they are not, and a blocked thread hangs a host whose own exit
   handler joins it. After the wait, on a thread other than the exiting one:
   - `llama_dart_exit_free` and the session free functions free nothing, and
     `llama_dart_exit_track` and `llama_dart_exit_untrack` return `false`;
   - a call in flight that begins returns the failure value of its function,
     with `the process is exiting` as `llama_dart_last_error`, and starts
     nothing: the exit no longer waits for it;
   - `llama_dart_exit_call_begin` counts nothing;
   - the call that the exit waited for returns to its caller.
5. While teardown waits, and from then on after an exit, a decode or an
   encode on the CPU backend ends with llama.cpp's status for an aborted
   evaluation (2). `llama_dart_init_from_model` sets the context's abort
   callback for that when the caller passes none, and the draft contexts of
   the speculative and MTP state get it too. Without it a long decode outlasts
   the wait and is still running when the exit reaches other libraries: with
   the BLAS module, OpenBLAS's destructor then never returns (see
   "Measurements"). llama.cpp logs three error lines for the ended decode.
6. libllamadart itself binds `__cxa_atexit` to the definition that Apple
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

- A Dart process in which another isolate runs Dart code while the exit
  waits for a call in flight. The Dart VM aborts
  (`runtime/vm/handles_impl.h: 39: error: unreachable code`,
  [llamadart#977](https://github.com/leehack/llamadart/issues/977)) when an
  isolate collects garbage while C `exit()` is taking its time, whatever
  takes it: a 300 ms exit handler with no llama.cpp in the process does it
  10 of 10. The wait is what fixes the crash, so it stays, and it is as short
  as the call: see "Dart" for the rates.
- A call in flight that outlasts the wait and cannot be ended: a decode on a
  GPU backend, which does not read the abort callback, an image or audio
  evaluation, or a context with an abort callback of its own. The exit then
  goes on with the call running, two seconds later than it would have, and
  what the call uses of other libraries may be gone. The lavapipe decode of
  1500 tokens in "Measurements" is such a call.
- An exit handler that another library registers after the two handlers above
  runs before the wait, under the calls in flight. A GPU driver that
  registers one late, during a decode for example, is in that position.
  Statics that llama.cpp creates late no longer matter.
- A call that is not a call in flight is not waited for, as on Apple. Its
  statics stay valid on Linux; what it uses of other libraries may not.
- `quick_exit` and `_exit` run no handler, as before. Neither destroys
  statics.
- A child of `fork` that calls `exit()` runs the handler with the registry it
  copied: it waits out the wait time for calls that are not its own.
- An exit handler that code inside a llama.cpp library registers with
  `atexit` binds to the same definition and is dropped. No backend built here
  registers one; a library such as the CUDA runtime, which the modules load
  as a library of its own, is not affected.
- The standard streams of a host built against libstdc++ older than GCC 13.
  There `<iostream>` gives every translation unit a static `ios_base::Init`,
  and `std::cout` is flushed when the last of them is destroyed. One inside a
  llama.cpp library is now never destroyed, so a host that turned off
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

- musl: not supported and not run. No bundle is built for it. The audit found
  that musl's `dlsym(RTLD_NEXT, "__cxa_atexit")` returns `NULL` in a library
  opened with `dlopen`, so libllamadart's own definition would register
  nothing there.
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
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 5 | 0 |
| | Vulkan | slow | 10 | 0 |
| inside one decode of 1500 tokens | CPU | fast | 1 | 0 |
| | CPU | slow | 10 | 0 |
| | CPU, BLAS | slow | 10 | 0 |
| | Vulkan | fast | 0 of 50 | 2 of 50 |
| | Vulkan | slow | 24 of 50 | 2 of 50 |
| 100 ms and 400 ms into the first load | CPU | slow | 9, 6 | 0, 0 |
| 200 ms and 800 ms into the first load | CPU | fast | 0, 0 | 0, 0 |
| 5 ms and 60 ms into the first load | Vulkan | fast | 0, 0 | 0, 0 |
| | Vulkan | slow | 10, 10 | 0, 0 |
| first load, once tensors are loading | CPU, Vulkan | fast, slow | 0 | 0 |
| halfway through a second load | CPU | slow | 1 | 0 |
| | CPU, Vulkan | fast | 0 | 0 |
| `return` from `main` while generating | CPU | fast | 8 | 0 |
| | Vulkan | fast | 3 | 0 |
| model idle; everything freed; `return` from `main` with a model idle | CPU, BLAS, Vulkan | fast, slow | 0 | 0 |

Every failed run ended in a signal: `SIGSEGV`, and on lavapipe before the
change also `SIGABRT` (`double free or corruption`). The scenarios that were
clean before are clean after, with the model left loaded and not freed.

One row is worse than before. The decode of 1500 tokens on lavapipe cannot be
ended: ggml-vulkan does not read the abort callback. It outlasts the two
seconds, and the exit then goes on under it. With a fast host that fails 2 of
50 (6 of 50 in the audit, with a backtrace in the LLVM compiler of lavapipe
under a thread that waits in `ggml_vk_flash_attn`), where it was clean 50 of
50 before, also when the exit came 2.3 s into the decode. Before, the
destructor of ggml-vulkan's device list destroyed the device at once; now the
device outlives the exit and the driver's own exit handlers meet a decode
that is still running. With a slow host the same row goes from 24 of 50 to 2
of 50. Whether a GPU driver behaves like lavapipe here is part of what is not
verified.

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

A host whose own exit handler stops a worker and joins it, with the worker
freeing its context and model through `llama_dart_exit_free`: clean 5 of 5
before and after. The first version of this change blocked that worker and
hung 5 of 5.

### Dart

The Dart VM aborts when one of its isolates collects garbage while C `exit()`
is under way
([llamadart#977](https://github.com/leehack/llamadart/issues/977)). A Dart
program with one isolate that decodes UTF-8 in a loop and an exit handler
that sleeps 300 ms, with no llama.cpp in the process, aborts 10 of 10. So
whatever the handler of libllamadart takes, it takes at that price, and it
takes time only while a guarded call is in flight.

Dart 3.13.1, CPU. `tests/manual/dart_exit_probe.dart` calls a guarded decode
and sample per step through a small C shim, on `stories15M`; llamadart rows
are its own `test/fixtures/llama_cpp_exit_probe.dart` and a variant with an
isolate that wakes on a 5 ms timer, on llamadart `9275de8` with the bundle
under test and Qwen3.5 0.8B Q4_0. The main isolate calls C `exit()` through
FFI. Failed runs of 10, or of 30 where it says so; "First version" is
`c9bc619`, which waited 250 ms at every exit and blocked the threads that
reached libllamadart afterwards.

| Exit | Other isolate | `v0.6.0-1` | First version | This rebuild |
| --- | --- | --- | --- | --- |
| llamadart `quit-generating` | none | 10 | 0 | 0 |
| llamadart `quit-loaded`, `quit-loading`, `return-loaded`, `kill-loading` | none | 0 | 0 | 0 |
| llamadart, right after a generation finished | 5 ms timer | 0 | 10 | 0 |
| llamadart, 1 s after a generation finished | 5 ms timer | 0 | 0 | 0 |
| llamadart, while generating | 5 ms timer | 6 | 10 | 0 |
| worker isolate generating | none | 13 of 30 | 0 | 0 of 30 |
| worker isolate generating | decoding in a loop | 13 of 30 | 10 | 3 of 30 |
| worker isolate inside one decode of 1500 tokens | decoding in a loop | 10 of 30 | 10 | 17 of 30 |
| right after a guarded call returned | decoding in a loop | 7 of 30 | 10 | 10 of 30 |
| worker isolate idle after a generation | 5 ms timer | 0 | 9 | 0 |
| worker isolate idle after a generation | decoding in a loop | 4 of 30 | 10 | 3 of 30 |
| model idle for 400 ms | decoding in a loop | 8 of 30 | 0 | 6 of 30 |
| `dart:io` `exit()` while generating; `main` returns while generating | none | 0 | 0 | 0 |

The failures of `v0.6.0-1` while generating are the `SIGSEGV` of the issue.
Those of this rebuild are all the abort of the Dart VM. An isolate that
decodes in a loop makes the VM abort in about a fifth of the exits with a
model loaded even where libllamadart has no handler at all (19 of 90 in the
three rows of `v0.6.0-1` with no call in flight), so the rows with one
compare rates: this rebuild is at the rate of `v0.6.0-1` where no call is in
flight (19 of 90), and below it while generating. It is above it in one row,
17 of 30 against 10 of 30: an exit inside a long decode while another isolate
keeps collecting garbage. The exit waits there until the decode has ended,
which takes as long as the tensor that is being computed, and the VM aborts
in that time.

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
- The same test source against the libraries of `v0.6.0-1`: `model-decode`,
  `model-late-load` and `model-decode-outlasts-wait` end in `SIGSEGV`, and
  `model-load` and `model-free-in-flight` report that the exit did not wait,
  5 of 5 each.
- Each part removed in turn fails a scenario:

  | Removed | Fails |
  | --- | --- |
  | the static destructor object | `model-decode-outlasts-wait` (`SIGSEGV`); `validate_linux_artifact.py` names seven libraries |
  | the handler at the first creating call | `model-load` |
  | the handler at the first tracked object | `exit-in-flight` |
  | the exit that frees nothing | `exit` |
  | the abort callback | `model-decode-ended` |
  | the return before the 250 ms | `model-exit-idle` (260 ms against a bound of 100 ms) |
  | threads not blocked after the exit | `exit-late-calls`, `model-exit-join-idle`, `model-exit-join-generating` |
  | the refusal of a late call | `exit-late-calls` |

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
  CPU, Vulkan and CUDA backends, at least 10 runs each, and an exit inside a
  prompt decode that takes longer than two seconds on each GPU backend, 50
  runs, against `v0.6.0-1`.
- Linux x64 by hand. The measurements are arm64; `validate_wrapper.yml` runs
  the scenarios on x64.
- HIP and musl builds, Android and Windows.
- Flutter. The Dart rows are the standalone VM.
