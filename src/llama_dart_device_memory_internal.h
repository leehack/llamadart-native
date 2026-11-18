#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Free GPU memory that counts other processes, for a Windows Vulkan or CUDA
// device. Both backends report a per-process budget there: WDDM does not
// deduct memory that other processes hold. Kept free of platform and llama.cpp
// headers so that the rules run in a plain test.

struct llama_dart_adapter_segment {
  uint64_t commit_limit = 0;
  uint64_t bytes_resident = 0;
  bool aperture = false;
};

// Free dedicated memory of an adapter from its memory segments. A discrete
// device counts the segments that are not aperture (host memory the GPU
// reaches), an integrated one counts all of them. Capacity is the smaller of
// the driver's total and the segments' commit limit; the driver's total is the
// capacity when the segments give none. False when no segment is counted.
static inline bool
llama_dart_adapter_free_bytes(const std::vector<llama_dart_adapter_segment> &segments,
                              uint64_t driver_total, bool integrated,
                              uint64_t *free_bytes) {
  uint64_t limit = 0;
  uint64_t resident = 0;
  bool counted = false;
  for (const llama_dart_adapter_segment &segment : segments) {
    if (segment.aperture && !integrated) {
      continue;
    }
    counted = true;
    limit += segment.commit_limit;
    resident += segment.bytes_resident;
  }
  if (!counted) {
    return false;
  }
  const uint64_t capacity =
      limit == 0 ? driver_total : std::min(driver_total, limit);
  *free_bytes = capacity > resident ? capacity - resident : 0;
  return true;
}

// The free memory of a device is what the driver reports or what the system
// does, whichever is smaller: the driver's number alone loses other
// processes' memory, and a wrong system number must not raise it.
static inline uint64_t llama_dart_effective_free(uint64_t driver_free,
                                                 uint64_t system_free) {
  return std::min(driver_free, system_free);
}

// ggml names its devices <prefix><index> (Vulkan0, CUDA1); the index is the
// backend's own ordinal. Anything else, such as the host buffers' Vulkan_Host
// or CUDA_Host, is not a device.
static inline bool llama_dart_device_index(const std::string &name,
                                           const std::string &prefix,
                                           int32_t *index) {
  if (name.size() <= prefix.size() ||
      name.compare(0, prefix.size(), prefix) != 0 ||
      name.size() - prefix.size() > 4) {
    return false;
  }
  int32_t value = 0;
  for (size_t i = prefix.size(); i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') {
      return false;
    }
    value = value * 10 + (name[i] - '0');
  }
  *index = value;
  return true;
}

static inline bool llama_dart_vulkan_device_index(const std::string &name,
                                                  int32_t *index) {
  return llama_dart_device_index(name, "Vulkan", index);
}

// Which API supplies the adapter LUID of a device whose system memory is read.
enum class llama_dart_memory_source { none, vulkan, cuda };

static inline llama_dart_memory_source
llama_dart_device_memory_source(const std::string &name, bool is_gpu) {
  int32_t index = 0;
  if (!is_gpu) {
    return llama_dart_memory_source::none;
  }
  if (name.compare(0, 6, "Vulkan") == 0) {
    return llama_dart_memory_source::vulkan;
  }
  if (llama_dart_device_index(name, "CUDA", &index)) {
    return llama_dart_memory_source::cuda;
  }
  return llama_dart_memory_source::none;
}

static inline std::string
llama_dart_unreadable_memory_message(const std::string &device,
                                     const std::string &detail) {
  return "cannot read the system-wide free memory of GPU device " + device +
         ": " + detail;
}

enum class llama_dart_system_read { not_applicable, ok, unavailable };

struct llama_dart_system_memory {
  llama_dart_system_read read = llama_dart_system_read::not_applicable;
  uint64_t free_bytes = 0;
  std::string detail;
};

// The CUDA driver calls that map a ggml CUDA device to its adapter LUID, as
// plain function pointers so that a test can fake them. All null when
// nvcuda.dll is not loaded.
struct llama_dart_cuda_api {
  int (*init)(unsigned flags) = nullptr;
  int (*device_count)(int *count) = nullptr;
  int (*device_get)(int *device, int ordinal) = nullptr;
  int (*device_name)(char *name, int length, int device) = nullptr;
  int (*device_luid)(char *luid, unsigned *node_mask, int device) = nullptr;
};

static inline std::string llama_dart_cuda_error(const char *call, int code) {
  return std::string(call) + " failed with CUresult " + std::to_string(code);
}

// The adapter LUID of the ggml device CUDA<index>. ggml-cuda numbers its
// devices like the driver API, so the index is the driver ordinal, unless
// GGML_CUDA_DEVICES splits a card into virtual devices: then the counts differ
// and the card's memory is not the device's, so the mapping is refused.
// cuInit is idempotent and makes the mapping independent of the caller's
// order of CUDA calls.
static inline bool llama_dart_cuda_device_luid(const llama_dart_cuda_api &api,
                                               int32_t index,
                                               int32_t registered_count,
                                               const char *description,
                                               uint8_t *luid,
                                               std::string *detail) {
  if (api.init == nullptr || api.device_count == nullptr ||
      api.device_get == nullptr || api.device_name == nullptr ||
      api.device_luid == nullptr) {
    *detail = "nvcuda.dll is not loaded or lacks the device calls";
    return false;
  }
  int code = api.init(0);
  if (code != 0) {
    *detail = llama_dart_cuda_error("cuInit", code);
    return false;
  }
  int count = 0;
  code = api.device_count(&count);
  if (code != 0) {
    *detail = llama_dart_cuda_error("cuDeviceGetCount", code);
    return false;
  }
  if (count != registered_count) {
    *detail = "the driver lists " + std::to_string(count) +
              " devices but ggml registers " + std::to_string(registered_count) +
              " (the driver and ggml count devices differently, e.g. from "
              "GGML_CUDA_DEVICES)";
    return false;
  }
  if (index < 0 || index >= count) {
    *detail = "the device index " + std::to_string(index) +
              " is outside the driver's " + std::to_string(count) + " devices";
    return false;
  }
  int device = 0;
  code = api.device_get(&device, index);
  if (code != 0) {
    *detail = llama_dart_cuda_error("cuDeviceGet", code);
    return false;
  }
  char name[256] = {};
  code = api.device_name(name, static_cast<int>(sizeof(name)), device);
  if (code != 0) {
    *detail = llama_dart_cuda_error("cuDeviceGetName", code);
    return false;
  }
  if (description == nullptr || std::string(name) != description) {
    *detail = "the CUDA driver lists a different device at its index";
    return false;
  }
  char raw[8] = {};
  unsigned node_mask = 0;
  code = api.device_luid(raw, &node_mask, device);
  if (code != 0) {
    *detail = llama_dart_cuda_error("cuDeviceGetLuid", code);
    return false;
  }
  bool any = false;
  for (char byte : raw) {
    any = any || byte != 0;
  }
  if (!any) {
    *detail = "the CUDA device has no adapter LUID (not a WDDM device)";
    return false;
  }
  for (size_t i = 0; i < 8; ++i) {
    luid[i] = static_cast<uint8_t>(raw[i]);
  }
  return true;
}

// The system's free memory of a device: not_applicable without a source,
// otherwise the adapter's free dedicated memory. `read_luid(luid, detail)`
// gives the adapter LUID by the source's API, `read_segments(luid, segments,
// detail)` the adapter's memory segments. Any failure is unavailable, with the
// detail the reader gave.
template <typename ReadLuid, typename ReadSegments>
static llama_dart_system_memory
llama_dart_system_memory_from(llama_dart_memory_source source,
                              uint64_t driver_total, bool integrated,
                              ReadLuid read_luid, ReadSegments read_segments) {
  llama_dart_system_memory result;
  if (source == llama_dart_memory_source::none) {
    return result;
  }
  result.read = llama_dart_system_read::unavailable;
  uint8_t luid[8] = {};
  if (!read_luid(luid, &result.detail)) {
    return result;
  }
  std::vector<llama_dart_adapter_segment> segments;
  if (!read_segments(luid, &segments, &result.detail)) {
    return result;
  }
  if (!llama_dart_adapter_free_bytes(segments, driver_total, integrated,
                                     &result.free_bytes)) {
    result.detail = "the adapter has no dedicated memory segment";
    return result;
  }
  result.read = llama_dart_system_read::ok;
  return result;
}

// Raises the margin of every device by what the driver's free memory exceeds
// the system's, so that the fit's `free - margin` is the effective free memory
// minus the caller's margin. `names` are the devices in the order of
// `margins`. A device whose system reading is needed but unavailable throws:
// the fit must not run on a number that ignores other processes.
template <typename ReadDriver, typename ReadSystem>
static void llama_dart_fit_margins(const std::vector<std::string> &names,
                                   size_t *margins, ReadDriver read_driver,
                                   ReadSystem read_system) {
  for (size_t i = 0; i < names.size(); ++i) {
    const llama_dart_system_memory system = read_system(i);
    if (system.read == llama_dart_system_read::not_applicable) {
      continue;
    }
    if (system.read == llama_dart_system_read::unavailable) {
      throw std::runtime_error(
          llama_dart_unreadable_memory_message(names[i], system.detail));
    }
    const uint64_t driver_free = read_driver(i);
    const uint64_t effective =
        llama_dart_effective_free(driver_free, system.free_bytes);
    margins[i] += static_cast<size_t>(driver_free - effective);
  }
}
