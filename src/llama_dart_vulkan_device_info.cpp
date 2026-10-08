#include "llama_dart_wrapper.h"

// ggml-vulkan is built for these platforms. Elsewhere libllamadart has no
// Vulkan backend to describe and never loads a Vulkan loader.
#if defined(_WIN32) || defined(__ANDROID__) || defined(__linux__)
#define LLAMA_DART_VULKAN_PROBE 1
#else
#define LLAMA_DART_VULKAN_PROBE 0
#endif

#if LLAMA_DART_VULKAN_PROBE

#include "llama_dart_vulkan_internal.h"

#include <mutex>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// The loader that ggml-vulkan links, found the way the system finds it for
// ggml-vulkan. libllamadart does not link it, so that it still loads where
// there is no Vulkan. The loader stays loaded: unloading one that has loaded
// drivers is not safe with every driver.
static PFN_vkGetInstanceProcAddr llama_dart_vulkan_open_loader() {
#if defined(_WIN32)
  const HMODULE loader = LoadLibraryW(L"vulkan-1.dll");
  return loader == nullptr
             ? nullptr
             : reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                   GetProcAddress(loader, "vkGetInstanceProcAddr"));
#else
#if defined(__ANDROID__)
  void *loader = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
#else
  void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
  return loader == nullptr
             ? nullptr
             : reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                   dlsym(loader, "vkGetInstanceProcAddr"));
#endif
}

struct llama_dart_vulkan_state {
  std::mutex mutex;
  bool probed = false;
  llama_dart_vulkan_devices devices;
};

// Probes once, as ggml-vulkan selects its devices once. Returns the result,
// which nothing changes afterwards.
static const llama_dart_vulkan_devices &llama_dart_vulkan_probed() {
  // Never destroyed: a thread may still read it after static destructors.
  static auto *state = new llama_dart_vulkan_state();
  std::lock_guard<std::mutex> lock(state->mutex);
  if (!state->probed) {
    llama_dart_vulkan_probe(llama_dart_vulkan_open_loader(),
                            getenv("GGML_VK_VISIBLE_DEVICES"), state->devices);
    state->probed = true;
  }
  return state->devices;
}

#endif

extern "C" {

LLAMADART_API int32_t llama_dart_vulkan_get_device_count(void) {
#if LLAMA_DART_VULKAN_PROBE
  try {
    const llama_dart_vulkan_devices &devices = llama_dart_vulkan_probed();
    return devices.status == LLAMA_DART_VULKAN_STATUS_OK ? devices.count
                                                         : devices.status;
  } catch (...) {
    return LLAMA_DART_VULKAN_STATUS_LOADER_ERROR;
  }
#else
  return LLAMA_DART_VULKAN_STATUS_UNSUPPORTED;
#endif
}

LLAMADART_API int32_t llama_dart_vulkan_get_device_info(
    int32_t index, struct llama_dart_vulkan_device_info *out_info) {
  if (out_info == nullptr || out_info->struct_size < sizeof(*out_info)) {
    return LLAMA_DART_VULKAN_STATUS_INVALID_ARGUMENT;
  }
#if LLAMA_DART_VULKAN_PROBE
  try {
    const llama_dart_vulkan_devices &devices = llama_dart_vulkan_probed();
    if (devices.status != LLAMA_DART_VULKAN_STATUS_OK) {
      return devices.status;
    }
    if (index < 0 || index >= devices.count) {
      return LLAMA_DART_VULKAN_STATUS_NO_DEVICE;
    }
    *out_info = devices.devices[index];
    return LLAMA_DART_VULKAN_STATUS_OK;
  } catch (...) {
    return LLAMA_DART_VULKAN_STATUS_LOADER_ERROR;
  }
#else
  (void)index;
  return LLAMA_DART_VULKAN_STATUS_UNSUPPORTED;
#endif
}
}
