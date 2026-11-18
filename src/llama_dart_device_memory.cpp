#include "llama_dart_device_memory.h"

#include "llama.h"
#include "llama-ext.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include "llama_dart_cuda_luid_internal.h"
#include "llama_dart_d3dkmt_internal.h"
#endif

llama_dart_system_memory llama_dart_read_system_memory(ggml_backend_dev_t dev) {
#if defined(_WIN32)
  // Through the properties, the one query that a device interface always has.
  ggml_backend_dev_props props = {};
  ggml_backend_dev_get_props(dev, &props);
  const enum ggml_backend_dev_type type = props.type;
  const std::string name = props.name == nullptr ? "" : props.name;
  const bool integrated = type == GGML_BACKEND_DEVICE_TYPE_IGPU;
  const llama_dart_memory_source source = llama_dart_device_memory_source(
      name, type == GGML_BACKEND_DEVICE_TYPE_GPU || integrated);
  const auto read_segments =
      [](const uint8_t *luid, std::vector<llama_dart_adapter_segment> *segments,
         std::string *detail) {
        return llama_dart_d3dkmt::read_segments(luid, *segments, *detail);
      };
  if (source == llama_dart_memory_source::cuda) {
    int32_t index = 0;
    llama_dart_device_index(name, "CUDA", &index);
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const int32_t registered =
        reg == nullptr ? 0 : static_cast<int32_t>(ggml_backend_reg_dev_count(reg));
    return llama_dart_system_memory_from(
        source, props.memory_total, integrated,
        [&](uint8_t *luid, std::string *detail) {
          return llama_dart_cuda_device_luid(llama_dart_cuda_resolve_api(),
                                             index, registered,
                                             props.description, luid, detail);
        },
        read_segments);
  }
  return llama_dart_system_memory_from(
      source, props.memory_total, integrated,
      [&](uint8_t *luid, std::string *detail) {
        int32_t index = 0;
        if (!llama_dart_vulkan_device_index(name, &index)) {
          *detail = "its name is not Vulkan<index>";
          return false;
        }
        char probed_name[256] = {};
        bool luid_valid = false;
        if (!llama_dart_vulkan_device_luid(index, probed_name,
                                           sizeof(probed_name), luid,
                                           &luid_valid)) {
          *detail = "the Vulkan probe has no such device";
          return false;
        }
        if (props.description == nullptr ||
            strcmp(probed_name, props.description) != 0) {
          *detail = "the Vulkan probe lists a different device at its index";
          return false;
        }
        if (!luid_valid) {
          *detail = "the Vulkan device has no adapter LUID";
          return false;
        }
        return true;
      },
      read_segments);
#else
  (void)dev;
  return llama_dart_system_memory();
#endif
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
      ggml_backend_dev_props props = {};
      ggml_backend_dev_get_props(dev, &props);
      throw std::runtime_error(llama_dart_unreadable_memory_message(
          props.name == nullptr ? "" : props.name, system.detail));
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
  // A host with no Vulkan or CUDA device has nothing to correct. The registry
  // decides, not the Vulkan probe: a CUDA host may have no usable Vulkan.
  bool corrected_backend = false;
  for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(ggml_backend_dev_get(i), &props);
    corrected_backend =
        corrected_backend ||
        llama_dart_device_memory_source(
            props.name == nullptr ? "" : props.name,
            props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) !=
            llama_dart_memory_source::none;
  }
  if (!corrected_backend) {
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
