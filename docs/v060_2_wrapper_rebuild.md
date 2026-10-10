# v0.6.0-2 wrapper rebuild: Linux host shutdown

llama.cpp remains v0.6.0 (`d81235049`). No wrapper export or header layout is
added. Android driver patches and image-token limits are separate work.

## Contract

Before a native host calls C `exit()`, it stops new work, stops and joins its
native workers, and shuts down its Dart isolates or Flutter engine. A kill
request is not completion: await each isolate's exit notification and native
finalizers before exiting. Free contexts before projectors / models. A native
call already in progress completes before its isolate can shut down.

Linux, Android excepted, retains the statics owned by the runtime bundle.
`src/llama_dart_static_destructors.c` supplies a hidden `__cxa_atexit` that drops
owned static destructors; external libraries keep their ordinary exit behavior.
The wrapper remains linked with `-z nodelete` so held objects and finalizer
callbacks remain valid across `dlclose`. Artifact validators reject missing
retention and owned imports of `__cxa_atexit` or the pre-GCC13 iostream initializer.

C exit registers no automatic Linux wrapper teardown or exit handler. It does
not set a late-call refusal state, flush streams early, wait, free live objects
or call `_exit`. Normal host callbacks, dependency destructors and libc stdio
flushing run in their normal order. An earlier native-host callback can still
join a worker and free its tracked objects; the wrapper does not park that
worker merely because libc exit began.

Apple automatic teardown and explicit `llama_dart_exit_teardown` are unchanged.
Do not invoke explicit teardown before joining workers: it parks later guarded
calls, so a subsequent host join can hang. Android is unchanged. Windows / musl
hardware qualification is not established by these Linux glibc measurements.

## Why exit must follow host shutdown

A direct C exit with live Dart isolates is unsafe independently of llama.cpp.
With an allocating isolate and an earlier300ms host callback, the VM aborts at
`handles_impl.h:39` without loading any native model (llamadart#977). Retaining
owned statics cannot protect a GPU driver's or BLAS library's destructors either.
Direct C exit while Dart isolates or dependency-using native workers are alive
is outside this contract, including calls from an evaluation callback or an
incomplete model load. CPU raw-exit controls remain diagnostics, not support
claims. A hung native call cannot be called quiescent after a timeout; the host
must withhold normal C exit until it completes or explicitly choose its own
abrupt process-termination policy.

The rejected candidate flushed streams and called `_exit` when a call was in
flight. It skipped earlier host callbacks, and its `fflush(NULL)` deadlocked if
a worker held a FILE lock whose earlier host callback would release it. Removing
only the flush lost buffered output. Both alternatives violate the accepted
preservation contract and are absent from this revision.

## Validation and replay

`llamadart_exit_teardown_exit-status` verifies exit37, the earlier callback,
buffered output, and a host callback joining a marked native worker that holds a
FILE lock. Its10s CTest timeout makes the old unconditional-flush deadlock fail.
`exit-late-calls` verifies the host callback can create and free tracked objects
without an automatic Linux refusal state. These are native compatibility
controls, without a running Dart VM. Apple teardown tests retain their contract.

The maintained manual `tests/manual/dart_host_shutdown_probe.dart` uses worker
NativeFinalizers, per-worker onExit notifications, an allocating isolate, a
slow native call and a slow earlier host callback. Build the adjacent shim as
in `dart_exit_probe.dart`; set `PROBE_STEP_DELAY_MS=300`, then run:

```bash
dart tests/manual/dart_host_shutdown_probe.dart libshim.so <bundle-dir> \
  cooperative-finalizer-slow-native <model.gguf>
```

Success requires exit0, `NATIVE_ACTIVE_BEFORE_KILL`, `NATIVE_CALLS_AFTER_EXIT 0`,
`NATIVE_FREE_COUNT_AFTER_EXIT 1`, `HOST_HANDLER_COMPLETED`, and `C_BUFFERED_PAYLOAD`.
`raw-control` reproduces the VM defect; `cooperative-control` verifies allocator
shutdown without loading a native bundle. Neither case requires model downloads.

Pre-implementation prototype evidence (not qualification of this revised head):
Linuxarm64 GCC14/glibc2.41, Dart3.13.1, stories15M CPU, five repeats per case /
build, released baseline vs rejected head vs preservation prototype.60/60
cooperative model cases and15/15 no-model cases retained callbacks / output.
30/30 active native-call cases waited314–339ms before onExit;15/15 finalizers
completed first.15/15 raw no-model controls aborted. The protocol also passed
on the released baseline, so it establishes the lifecycle boundary, not a new
native feature. The GCC13 dependencies were reused read-only; this differs
from Ubuntu24.04/GCC13 release qualification.

Current-head CI, independent audit and exact-runtime replay are required anew.
Remaining release gates include real GPU / OpenBLAS driver shutdown, image
runtime, Flutter Linux host ordering, final published artifacts and device
qualification. No earlier raw-exit result establishes those gates.
