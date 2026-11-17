#include "llama_dart_device_memory.h"

#include "llama.h"
#include "llama-ext.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include "llama_dart_d3dkmt_internal.h"
#endif

llama_dart_system_memory llama_dart_read_system_memory(ggml_backend_dev_t dev) {
  llama_dart_system_memory result;
#if defined(_WIN32)
  const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
  const std::string name = ggml_backend_dev_name(dev);
  if (type != GGML_BACKEND_DEVICE_TYPE_GPU &&
      type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
    return result;
  }
  if (name.compare(0, 6, "Vulkan") != 0) {
    return result;
  }
  result.read = llama_dart_system_read::unavailable;
  int32_t index = 0;
  if (!llama_dart_vulkan_device_index(name, &index)) {
    result.detail = "its name is not Vulkan<index>";
    return result;
  }
  char probed_name[256] = {};
  uint8_t luid[8] = {};
  bool luid_valid = false;
  if (!llama_dart_vulkan_device_luid(index, probed_name, sizeof(probed_name),
                                     luid, &luid_valid)) {
    result.detail = "the Vulkan probe has no such device";
    return result;
  }
  if (strcmp(probed_name, ggml_backend_dev_description(dev)) != 0) {
    result.detail = "the Vulkan probe lists a different device at its index";
    return result;
  }
  if (!luid_valid) {
    result.detail = "the Vulkan device has no adapter LUID";
    return result;
  }
  std::vector<llama_dart_adapter_segment> segments;
  if (!llama_dart_d3dkmt::read_segments(luid, segments, result.detail)) {
    return result;
  }
  size_t driver_free = 0;
  size_t driver_total = 0;
  ggml_backend_dev_memory(dev, &driver_free, &driver_total);
  if (!llama_dart_adapter_free_bytes(segments, driver_total,
                                     type == GGML_BACKEND_DEVICE_TYPE_IGPU,
                                     &result.free_bytes)) {
    result.detail = "the adapter has no dedicated memory segment";
    return result;
  }
  result.read = llama_dart_system_read::ok;
#else
  (void)dev;
#endif
  return result;
}

size_t llama_dart_device_free_bytes(ggml_backend_dev_t dev, size_t driver_free,
                                    bool strict) {
  const llama_dart_system_memory system = llama_dart_read_system_memory(dev);
  switch (system.read) {
  case llama_dart_system_read::ok:
    return static_cast<size_t>(
        llama_dart_effective_free(driver_free, system.free_bytes));
  case llama_dart_system_read::unavailable:
    if (strict) {
      throw std::runtime_error(llama_dart_unreadable_memory_message(
          ggml_backend_dev_name(dev), system.detail));
    }
    return driver_free;
  case llama_dart_system_read::not_applicable:
    break;
  }
  return driver_free;
}

void llama_dart_fit_apply_system_margins(const char *path_model,
                                         const llama_model_params &mparams,
                                         size_t *margins, size_t n_margins) {
#if defined(_WIN32)
  // A host with no Vulkan device has nothing to correct.
  char probed_name[8];
  uint8_t probed_luid[8];
  bool probed_valid = false;
  if (!llama_dart_vulkan_device_luid(0, probed_name, sizeof(probed_name),
                                     probed_luid, &probed_valid)) {
    return;
  }
  // The model's device list comes before the vocabulary-only branch of the
  // loader, so this load reads no weights and creates no buffers.
  llama_model_params probe = mparams;
  probe.vocab_only = true;
  probe.progress_callback = nullptr;
  probe.check_tensors = false;
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_model_load_from_file(path_model, probe), &llama_model_free);
  if (model == nullptr) {
    // The fit reports an unreadable model as its own error.
    return;
  }
  const int32_t n_devices = llama_model_n_devices(model.get());
  if (n_devices < 0 || static_cast<size_t>(n_devices) > n_margins) {
    throw std::runtime_error("unexpected device count for the memory margins");
  }
  std::vector<ggml_backend_dev_t> devices;
  std::vector<std::string> names;
  std::vector<size_t> driver_free;
  for (int32_t i = 0; i < n_devices; ++i) {
    ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    devices.push_back(dev);
    names.push_back(ggml_backend_dev_name(dev));
    driver_free.push_back(free_bytes);
  }
  llama_dart_fit_margins(
      names, margins, [&](size_t i) { return uint64_t(driver_free[i]); },
      [&](size_t i) { return llama_dart_read_system_memory(devices[i]); });
#else
  (void)path_model;
  (void)mparams;
  (void)margins;
  (void)n_margins;
#endif
}
