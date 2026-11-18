#pragma once

#ifndef _WIN32
#error "The nvcuda.dll lookup is Windows only"
#endif

#include "llama_dart_device_memory_internal.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

// llamadart.dll does not link CUDA and ggml's device properties carry no
// adapter LUID, so the driver calls are taken from nvcuda.dll when ggml-cuda
// has loaded it. GetModuleHandleW never loads a library, so no search path is
// involved: an absent module means no CUDA device to read.
inline llama_dart_cuda_api llama_dart_cuda_resolve_api() {
  llama_dart_cuda_api api;
  const HMODULE module = GetModuleHandleW(L"nvcuda.dll");
  if (module == nullptr) {
    return api;
  }
  const auto resolve = [&](const char *name) {
    return reinterpret_cast<void *>(GetProcAddress(module, name));
  };
  api.init = reinterpret_cast<decltype(api.init)>(resolve("cuInit"));
  api.device_count =
      reinterpret_cast<decltype(api.device_count)>(resolve("cuDeviceGetCount"));
  api.device_get =
      reinterpret_cast<decltype(api.device_get)>(resolve("cuDeviceGet"));
  api.device_name =
      reinterpret_cast<decltype(api.device_name)>(resolve("cuDeviceGetName"));
  api.device_luid =
      reinterpret_cast<decltype(api.device_luid)>(resolve("cuDeviceGetLuid"));
  return api;
}
