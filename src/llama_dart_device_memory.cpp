#include "llama_dart_device_memory.h"

#include "llama.h"
#include "llama-ext.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include "llama_dart_cuda_luid_internal.h"
#include "llama_dart_d3dkmt_internal.h"
#endif

#if defined(LLAMADART_TESTING)
#include "llama_dart_fit_test_hooks.h"
#endif

#if defined(LLAMADART_TESTING) && defined(_WIN32)
static const llama_dart_test_win_dev_config *g_test_win_dev = nullptr;

extern "C" void llama_dart_test_set_win_dev_config(
    const llama_dart_test_win_dev_config *config) {
  g_test_win_dev = config;
}
#endif

static ggml_backend_dev_props
llama_dart_system_memory_props(ggml_backend_dev_t dev) {
  ggml_backend_dev_props props = {};
  if (dev == nullptr) {
    throw std::runtime_error("missing backend device for memory measurement");
  }
#if defined(_WIN32)
  ggml_backend_dev_get_props(dev, &props);
#endif
  return props;
}

static llama_dart_system_memory
llama_dart_read_system_memory(ggml_backend_dev_t dev,
                              const ggml_backend_dev_props &props) {
#if defined(_WIN32)
  const enum ggml_backend_dev_type type = props.type;
  const std::string name = props.name == nullptr ? "" : props.name;
  const bool integrated = type == GGML_BACKEND_DEVICE_TYPE_IGPU;
  const llama_dart_memory_source source = llama_dart_device_memory_source(
      name, type == GGML_BACKEND_DEVICE_TYPE_GPU || integrated);
  const auto read_segments =
      [dev](const uint8_t *luid, std::vector<llama_dart_adapter_segment> *segments,
            std::string *detail) {
#if defined(LLAMADART_TESTING)
        if (g_test_win_dev != nullptr &&
            (dev == g_test_win_dev->cuda_device ||
             dev == g_test_win_dev->vulkan_device)) {
          return llama_dart_d3dkmt::read_segments_with(
              luid, g_test_win_dev->kmt_query, *segments, *detail);
        }
#endif
        return llama_dart_d3dkmt::read_segments(luid, *segments, *detail);
      };
  if (source == llama_dart_memory_source::cuda) {
    int32_t index = 0;
    llama_dart_device_index(name, "CUDA", &index);
    llama_dart_cuda_api cuda_api;
    int32_t registered = 0;
#if defined(LLAMADART_TESTING)
    if (g_test_win_dev != nullptr && dev == g_test_win_dev->cuda_device) {
      cuda_api.init = g_test_win_dev->cuda_init;
      cuda_api.device_count = g_test_win_dev->cuda_device_count;
      cuda_api.device_get = g_test_win_dev->cuda_device_get;
      cuda_api.device_name = g_test_win_dev->cuda_device_name;
      cuda_api.device_luid = g_test_win_dev->cuda_device_luid;
      registered = g_test_win_dev->cuda_registered_count;
    } else
#endif
    {
      cuda_api = llama_dart_cuda_resolve_api();
      ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
      if (reg != nullptr) {
        registered = static_cast<int32_t>(ggml_backend_reg_dev_count(reg));
      }
    }
    return llama_dart_system_memory_from(
        source, props.memory_total, integrated,
        [&](uint8_t *luid, std::string *detail) {
          return llama_dart_cuda_device_luid(cuda_api, index, registered,
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
#if defined(LLAMADART_TESTING)
        if (g_test_win_dev != nullptr && dev == g_test_win_dev->vulkan_device) {
          std::memcpy(luid, g_test_win_dev->vulkan_luid, sizeof(g_test_win_dev->vulkan_luid));
          return true;
        }
#endif
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
  (void)props;
  return llama_dart_system_memory();
#endif
}

llama_dart_system_memory llama_dart_read_system_memory(ggml_backend_dev_t dev) {
  return llama_dart_read_system_memory(dev, llama_dart_system_memory_props(dev));
}

size_t llama_dart_device_free_bytes(ggml_backend_dev_t dev, size_t driver_free,
                                    bool strict) {
  const ggml_backend_dev_props props = llama_dart_system_memory_props(dev);
  const llama_dart_system_memory system = llama_dart_read_system_memory(dev, props);
  switch (system.read) {
  case llama_dart_system_read::ok:
    return static_cast<size_t>(
        llama_dart_effective_free(driver_free, system.free_bytes));
  case llama_dart_system_read::unavailable:
    if (strict) {
      throw std::runtime_error(llama_dart_unreadable_memory_message(
          props.name == nullptr ? "" : props.name, system.detail));
    }
    return driver_free;
  case llama_dart_system_read::not_applicable:
    break;
  }
  return driver_free;
}

#if defined(_WIN32) || defined(LLAMADART_TESTING)
// Other threads can log while a fit holds the wrapper's log mutex. Keep the
// forwarding callback in static storage and filter only the probing thread.
static std::atomic<ggml_log_callback> g_fit_probe_logger{nullptr};
static thread_local bool g_fit_probe_scope_active = false;
static thread_local ggml_log_level g_fit_probe_level = GGML_LOG_LEVEL_NONE;

static void llama_dart_forward_fit_probe_log(ggml_log_level level,
                                             const char *text,
                                             void *user_data) {
  if (g_fit_probe_scope_active) {
    if (level != GGML_LOG_LEVEL_CONT) {
      g_fit_probe_level = level;
    }
    if (g_fit_probe_level == GGML_LOG_LEVEL_INFO) {
      return;
    }
  }
  if (const auto callback = g_fit_probe_logger.load()) {
    callback(level, text, user_data);
  }
}

class llama_dart_fit_probe_log_scope {
public:
  llama_dart_fit_probe_log_scope() {
    llama_log_get(&callback_, &user_data_);
    g_fit_probe_logger.store(callback_);
    g_fit_probe_level = GGML_LOG_LEVEL_NONE;
    g_fit_probe_scope_active = true;
    llama_log_set(llama_dart_forward_fit_probe_log, user_data_);
  }

  ~llama_dart_fit_probe_log_scope() {
    llama_log_set(callback_, user_data_);
    g_fit_probe_scope_active = false;
  }

  llama_dart_fit_probe_log_scope(const llama_dart_fit_probe_log_scope &) =
      delete;
  llama_dart_fit_probe_log_scope &
  operator=(const llama_dart_fit_probe_log_scope &) = delete;

private:
  ggml_log_callback callback_ = nullptr;
  void *user_data_ = nullptr;
};

static llama_model *
llama_dart_load_fit_probe(const char *path_model,
                          const llama_model_params &mparams) {
  // The load gets the selected devices in margin order without reading weights.
  llama_model_params probe = mparams;
  probe.vocab_only = true;
  probe.progress_callback = nullptr;
  probe.check_tensors = false;
  const llama_dart_fit_probe_log_scope log_scope;
  return llama_model_load_from_file(path_model, probe);
}
#endif

#if defined(LLAMADART_TESTING)
extern "C" llama_model *
llama_dart_test_load_fit_probe(const char *path_model,
                               const llama_model_params *mparams) {
  return llama_dart_load_fit_probe(path_model, *mparams);
}
#endif

void llama_dart_fit_apply_system_margins(const char *path_model,
                                         const llama_model_params &mparams,
                                         size_t *margins, size_t n_margins) {
#if defined(_WIN32)
  // A registered Vulkan or CUDA GPU can require margin correction even in a
  // CUDA-only bundle. The probe below resolves the model's selected devices.
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
  std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
      llama_dart_load_fit_probe(path_model, mparams), &llama_model_free);
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
