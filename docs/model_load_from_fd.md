# Model load from a caller-owned file descriptor

`llama_dart_model_load_from_fd` and `llama_dart_owner_model_load_from_fd` load a
GGUF through a descriptor the caller opened, for
[llamadart#243](https://github.com/leehack/llamadart/issues/243) (no-copy
loading of user-granted files). Tracking issue:
[#237](https://github.com/leehack/llamadart-native/issues/237). The wrappers
are the descriptor twins of `llama_dart_model_load_from_file`: the same exit
teardown guard, progress callback, cancellation and owner semantics.

```c
uint32_t llama_dart_model_fd_api_version(void);   // 0 on Windows, 1 elsewhere
int32_t  llama_dart_model_load_from_fd(
    int32_t fd, int64_t size, struct llama_model_params params,
    struct llama_model ** out_model);
int32_t  llama_dart_owner_model_load_from_fd(
    void * owner, int32_t fd, int64_t size, struct llama_model_params params,
    struct llama_model ** out_model);
```

A binding checks `llama_dart_model_fd_api_version() >= 1` before use and keeps
its path-based loader as the fallback.

## Ownership

- The caller keeps `fd`. The wrapper never closes it and never keeps it: it
  `dup`s with `F_DUPFD_CLOEXEC`, wraps the duplicate with `fdopen`, calls
  `llama_model_load_from_file_ptr`, and `fclose`s the duplicate before it
  returns. After the call the model holds no descriptor, so the caller may
  close `fd` at once (Android: close the `ParcelFileDescriptor`).
- The duplicate shares the caller's file offset. The wrapper reads from offset
  0 and restores the offset it found before returning. Do not use `fd` from
  another thread during the call.
- With an mmap load mode (the default `LLAMA_LOAD_MODE_AUTO`/`MMAP`) the model's
  tensors stay mapped from the file after the descriptor is gone: the kernel
  keeps the mapping's file reference. Nothing is copied into anonymous memory
  and pages are file-backed. The file must not be truncated or rewritten while
  the model is alive (a later access faults). Deleting or renaming it is safe.
- With `LLAMA_LOAD_MODE_NONE` the tensors are read through the stream into
  memory the model owns: a resident copy, still no copy on disk. Use it for
  descriptors that cannot be mapped (mmap failure is `LOAD_FAILED`, not a silent
  fallback).
- `*out_model` is set on every call: the tracked model on `SUCCESS`, `NULL`
  otherwise. Free the model with `llama_dart_exit_free`, as for the path loader.

## Status codes

Every status other than `SUCCESS` sets `llama_dart_last_error` (read it from the
same thread) and leaves the caller's descriptor open and at its offset.

| Value | Name | Cause |
| --- | --- | --- |
| 0 | `SUCCESS` | `*out_model` is a tracked model |
| 1 | `INVALID_ARGUMENT` | `out_model` is `NULL` |
| 2 | `UNSUPPORTED` | Windows: no descriptor loader. Never a success |
| 3 | `INVALID_DESCRIPTOR` | negative, not open, write-only, not a regular file (directory, pipe, socket), not seekable |
| 4 | `SIZE_MISMATCH` | `size >= 0` and different from `fstat` size |
| 5 | `LOAD_FAILED` | llama.cpp refused it: not a GGUF, truncated, unsupported, mmap failed, or cancelled by teardown |
| 6 | `ERROR` | caught exception, unknown owner or inactive owner, no free descriptor |

`size` is the length the caller expects (its own stat, or the content
provider's `AssetFileDescriptor.getLength()`); pass a negative value to skip the
check.

## Scope

- One GGUF from offset 0. Split GGUFs and a GGUF embedded at an offset are not
  supported by this entry point.
- The model has no path. Path-only helpers (`llama_dart_fit_params`, a
  projector loaded with `llama_dart_mtmd_init_from_file`) still take paths.
- Windows returns `UNSUPPORTED` for every call that passes argument validation.

## Platform notes

- **Android.** `ContentResolver.openFileDescriptor(uri, "r")` on a
  storage-backed document is expected to return a regular, seekable descriptor,
  which this accepts (not yet run against a real provider). A provider that streams the document (a pipe, or a
  `ProxyFileDescriptorCallback`) returns `INVALID_DESCRIPTOR`; the app falls
  back to copying the file into its cache. Pass `ParcelFileDescriptor.getFd()`
  and keep the object open until the call returns, or `detachFd()` and close it
  yourself.
- **iOS and macOS.** Open the file between
  `startAccessingSecurityScopedResource` and `stopAccessingSecurityScopedResource`
  and hold the scope until the call returns. The wrapper needs no path and no
  scope after that; that an already open descriptor and its mapping outlive the
  scope on a device is not yet verified.

## Tests

`llamadart_model_fd_contract` covers the refusals (negative, closed, directory,
pipe, write-only, not a GGUF, empty, size mismatch, inactive owner) without a
model. `llamadart_model_fd_model` loads the generated test model in both load
modes from a descriptor at a non-zero offset, closes the descriptor, and
compares the logits with a path load; it also checks that the descriptor and
offset survive success and failure and that owner release frees the model. On
Windows the contract test asserts `UNSUPPORTED`.
