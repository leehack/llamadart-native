# Creation-time worker ownership

The additive owner API v1 prepares the native repair for
[llamadart #1403](https://github.com/leehack/llamadart/issues/1403). It does not
publish a runtime, repin core or remove legacy exports.

A pending Dart isolate kill takes effect after a synchronous creator returns,
before Dart can attach an object finalizer. Models, contexts and projectors
therefore need ownership established inside their native creator. Owner-aware
creation publishes into the existing exit registry, under its existing object
identity, before returning. Dependent speculative/MTP and TTS creators inherit
the owner from their tracked inputs, and reject cross-owner dependencies.

## Bootstrap and lifetime

| State | Admission | Parent bootstrap cleanup | Worker cleanup |
| --- | --- | --- | --- |
| Pending | No payload creation | Closes the empty token | May close it |
| Active | Admits creators with same-owner dependencies | Does nothing | Closes after worker exit or cooperative disposal |
| Closing | Refuses new creators; admitted creators finish or free | No additional effect | Waits for admitted creators and frees |
| Closed | Token lookup fails | No-op | No-op |

The parent creates the pending token and attaches its bootstrap-abort finalizer
before spawning the killable worker. The worker attaches one stable native
release finalizer to an isolate-root anchor, then activates the token. No
payload may be created between token receipt and that attachment. A kill before
attachment leaves no payload; the live parent's onExit handler closes the empty
token. The parent also drains after actual worker exit, on a cleanup isolate,
and awaits that drain as part of the worker's recorded exit.

The parent finalizer never drains active payload: a parent can disappear while
its worker still runs. The worker's stable finalizer handles orderly
isolate-group shutdown; Dart promises that boundary, rather than prompt native
finalization at individual onExit. Ordinary host shutdown must still stop and
join workers and native finalizers before C exit.

Tokens are opaque, never-reused integer identities encoded as `void *`; they
are never dereferenced. Closing removes their control record after its payload
drains. In-flight native users retain the record with `shared_ptr`; late or
duplicate finalizers find no token and cannot touch reused memory. A parent
killed between creating an empty token and attaching its bootstrap finalizer
can leave a small empty control record until process termination. This is
separate from the model/context payload guarantee; no claim covers arbitrary
simultaneous parent/worker/process termination.

## Shared cleanup

Owner entries use the process registry's stage and reverse-creation ordering.
The owner waits for admitted creators and ordinary frees. Once closing starts,
later ordinary frees cannot steal a model while an earlier context callback is
in flight. Each entry is erased exactly once under the registry mutex; cleanup
callbacks run outside it. Reentrant close from a creator/free callback requests
closure without waiting on itself. The outermost scope completes the drain.
A process teardown racing an owner drain waits for its guarded call; reentrant
owner release from process teardown leaves the process snapshot in charge.

Existing process-tracked entries retain their original callback, stage and
dependencies when adopted. Raw held session objects can be adopted too, so
batches, samplers and decision-head resources already registered with the
service drain before their models. Adoption does not close a raw allocation's
return-to-adoption gap. Broader #629 work still includes token-buffer and
sampler-child allocation, decision-head backend/buffer/scheduler and ggml
contexts, LoRA/temporary allocations and other objects not registered by the
existing service. These are not universal reclamation guarantees.

`llama_dart_owner_release` must not race ordinary inference: use it only after
the worker stops or cooperative service disposal has stopped its operations.
It may race admitted creators, ordinary frees, duplicate release and process
teardown. Older consumers retain their previous exports and behavior.

## Validation

Configure a Debug build with `LLAMADART_BUILD_TESTS=ON`, then build
`llamadart_owner_api_test`, `llamadart_exit_teardown_test` and
`llamadart_speculative_api_test`. Run:

```sh
ctest --test-dir build/owner -R 'llamadart_owner|llamadart_speculative_api_test' \
  --output-on-failure
```

The tests cover bootstrap refusal/abandonment, independent owners, raw adoption,
foreign dependencies, ordinary/duplicate/reentrant frees, stage ordering,
creation failure, actual model/context/session creation, admitted-creator
closure, ordinary free during owner drain, and process teardown races.
The test-only `llamadart_owner_creation_gate` library pauses after calling the
real creator, using function pointers from the exact tested wrapper image.
Core's local-only `llama_cpp_owner_creation_e2e_test.dart` queues a real worker
kill there before allowing FFI to return, for both models and late contexts.
It also checks the stable finalizer at separate-isolate-group shutdown without
first invoking the parent's explicit drain.

Real model-backed kill-loading, idle/generation kills and ordinary disposal
remain separate qualification controls. Record the source diff/commit, native
library hash, model hash, actual backend and metrics. In particular, `dart test`
runs build hooks that can replace a manually copied candidate with the pinned
runtime. Use a supported local-bundle override or a pinned direct `dartvm`
package:test launcher with the already-prepared native-assets mapping, and
verify the loaded file's hash before and after the run. A passing unrelated
model is not broad hardware or model-family qualification.
