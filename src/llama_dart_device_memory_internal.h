#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Free GPU memory that counts other processes, for a Windows Vulkan device.
// ggml-vulkan reports a per-process budget there: WDDM does not deduct memory
// that other processes hold. Kept free of platform and llama.cpp headers so
// that the rules run in a plain test.

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

// ggml-vulkan names its devices Vulkan<index>; the index is the position in
// the probe's selection. Anything else, such as the host buffer's
// Vulkan_Host, is not a device.
static inline bool llama_dart_vulkan_device_index(const std::string &name,
                                                  int32_t *index) {
  static const std::string prefix = "Vulkan";
  if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0 ||
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
