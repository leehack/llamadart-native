#include "llama_dart_device_memory.h"

#include "llama.h"
#include "llama-ext.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#if defined(_WIN32)
#include "llama_dart_cuda_luid_internal.h"
#include "llama_dart_d3dkmt_internal.h"
#include <psapi.h>
#endif

#if defined(LLAMADART_TESTING)
#include "llama_dart_fit_test_hooks.h"
#endif

#if defined(LLAMADART_TESTING) && defined(_WIN32)
namespace {
struct llama_dart_test_win_dev_state {
  bool enabled = false;
  std::string name;
  std::string description;
  enum ggml_backend_dev_type type = GGML_BACKEND_DEVICE_TYPE_CPU;
  size_t memory_free = 0;
  size_t memory_total = 0;
  bool has_cuda_api = false;
  llama_dart_cuda_api cuda_api;
  int32_t cuda_registered_count = 0;
  llama_dart_d3dkmt::QueryStatistics kmt_query = nullptr;
};

llama_dart_test_win_dev_state g_test_win_dev;
} // namespace

extern "C" void llama_dart_test_set_win_dev_config(
    const llama_dart_test_win_dev_config *config) {
  g_test_win_dev = {};
  if (config == nullptr || !config->enabled) {
    return;
  }
  g_test_win_dev.enabled = true;
  g_test_win_dev.name = config->name != nullptr ? config->name : "";
  g_test_win_dev.description =
      config->description != nullptr ? config->description : "";
  g_test_win_dev.type = static_cast<enum ggml_backend_dev_type>(config->type);
  g_test_win_dev.memory_free = config->memory_free;
  g_test_win_dev.memory_total = config->memory_total;
  g_test_win_dev.has_cuda_api = config->has_cuda_api;
  g_test_win_dev.cuda_api.init = config->cuda_init;
  g_test_win_dev.cuda_api.device_count = config->cuda_device_count;
  g_test_win_dev.cuda_api.device_get = config->cuda_device_get;
  g_test_win_dev.cuda_api.device_name = config->cuda_device_name;
  g_test_win_dev.cuda_api.device_luid = config->cuda_device_luid;
  g_test_win_dev.cuda_registered_count = config->cuda_registered_count;
  g_test_win_dev.kmt_query =
      reinterpret_cast<llama_dart_d3dkmt::QueryStatistics>(config->kmt_query);
}

bool llama_dart_test_win_dev_override_memory(ggml_backend_dev_t dev,
                                             size_t *free, size_t *total) {
  (void)dev;
  if (!g_test_win_dev.enabled) {
    return false;
  }
  *free = g_test_win_dev.memory_free;
  *total = g_test_win_dev.memory_total;
  return true;
}

bool llama_dart_test_win_dev_override_props(
    ggml_backend_dev_t dev, struct ggml_backend_dev_props *props) {
  (void)dev;
  if (!g_test_win_dev.enabled) {
    return false;
  }
  *props = {};
  props->name = g_test_win_dev.name.c_str();
  props->description = g_test_win_dev.description.c_str();
  props->type = g_test_win_dev.type;
  props->memory_free = g_test_win_dev.memory_free;
  props->memory_total = g_test_win_dev.memory_total;
  return true;
}
#endif

static ggml_backend_dev_props
llama_dart_system_memory_props(ggml_backend_dev_t dev) {
  ggml_backend_dev_props props = {};
#if defined(LLAMADART_TESTING) && defined(_WIN32)
  if (llama_dart_test_win_dev_override_props(dev, &props)) {
    return props;
  }
#endif
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
      [](const uint8_t *luid, std::vector<llama_dart_adapter_segment> *segments,
         std::string *detail) {
#if defined(LLAMADART_TESTING)
        if (g_test_win_dev.enabled) {
          return llama_dart_d3dkmt::read_segments_with(
              luid, g_test_win_dev.kmt_query, *segments, *detail);
        }
#endif
        return llama_dart_d3dkmt::read_segments(luid, *segments, *detail);
      };
  if (source == llama_dart_memory_source::cuda) {
    int32_t index = 0;
    llama_dart_device_index(name, "CUDA", &index);
#if defined(LLAMADART_TESTING)
    const llama_dart_cuda_api cuda_api = g_test_win_dev.enabled &&
                                                 g_test_win_dev.has_cuda_api
                                             ? g_test_win_dev.cuda_api
                                             : llama_dart_cuda_resolve_api();
    const int32_t registered =
        g_test_win_dev.enabled
            ? g_test_win_dev.cuda_registered_count
            : [&] {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                return reg == nullptr
                           ? 0
                           : static_cast<int32_t>(ggml_backend_reg_dev_count(reg));
              }();
#else
    const llama_dart_cuda_api cuda_api = llama_dart_cuda_resolve_api();
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const int32_t registered =
        reg == nullptr ? 0 : static_cast<int32_t>(ggml_backend_reg_dev_count(reg));
#endif
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

bool llama_dart_fit_apply_system_margins(const char *path_model,
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
    return false;
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
    return false;
  }
  const int32_t n_devices = llama_model_n_devices(model.get());
  if (n_devices < 0 || static_cast<size_t>(n_devices) > n_margins) {
    throw std::runtime_error("unexpected device count for the memory margins");
  }
  std::vector<ggml_backend_dev_t> devices;
  std::vector<std::string> names;
  std::vector<size_t> driver_free;
  bool has_wddm_cuda = false;
  for (int32_t i = 0; i < n_devices; ++i) {
    ggml_backend_dev_t dev = llama_model_get_device(model.get(), i);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    const std::string name = props.name == nullptr ? "" : props.name;
    const bool is_gpu = props.type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                        props.type == GGML_BACKEND_DEVICE_TYPE_IGPU;
    if (llama_dart_device_memory_source(name, is_gpu) ==
        llama_dart_memory_source::cuda) {
      has_wddm_cuda = true;
    }
    devices.push_back(dev);
    names.push_back(name);
    driver_free.push_back(free_bytes);
  }
  llama_dart_fit_margins(
      names, margins, [&](size_t i) { return uint64_t(driver_free[i]); },
      [&](size_t i) { return llama_dart_read_system_memory(devices[i]); });
  return has_wddm_cuda;
#else
  (void)path_model;
  (void)mparams;
  (void)margins;
  (void)n_margins;
  return false;
#endif
}

uint64_t llama_dart_read_system_commit_available_bytes() {
#if defined(_WIN32)
  using GetPerfInfoFn = BOOL(WINAPI *)(PPERFORMANCE_INFORMATION, DWORD);
  const HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
  const auto get_perf_info =
      kernel32 == nullptr
          ? nullptr
          : reinterpret_cast<GetPerfInfoFn>(
                GetProcAddress(kernel32, "K32GetPerformanceInfo"));
  if (get_perf_info == nullptr) {
    throw std::runtime_error(
        "cannot read Windows system commit headroom: kernel32.dll has no "
        "K32GetPerformanceInfo");
  }
  PERFORMANCE_INFORMATION info = {};
  info.cb = sizeof(info);
  if (!get_perf_info(&info, sizeof(info))) {
    throw std::runtime_error(
        "cannot read Windows system commit headroom: K32GetPerformanceInfo "
        "failed with error " +
        std::to_string(GetLastError()));
  }
  const llama_dart_commit_headroom headroom = llama_dart_commit_available_bytes(
      static_cast<uint64_t>(info.CommitLimit),
      static_cast<uint64_t>(info.CommitTotal),
      static_cast<uint64_t>(info.PageSize));
  if (!headroom.ok) {
    throw std::runtime_error("cannot read Windows system commit headroom: " +
                             headroom.detail);
  }
  return headroom.available_bytes;
#else
  throw std::runtime_error(
      "Windows system commit headroom is only available on Windows");
#endif
}

#if defined(LLAMADART_TESTING)
extern "C" uint64_t
llama_dart_test_read_system_commit_available_bytes(void) {
  return llama_dart_read_system_commit_available_bytes();
}
#endif
