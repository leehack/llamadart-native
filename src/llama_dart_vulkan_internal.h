#pragma once

#include "llama_dart_wrapper.h"

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

// The devices that ggml-vulkan selects, in its order, as
// llama_dart_vulkan_get_device_info reports them.
struct llama_dart_vulkan_devices {
  // ggml-vulkan holds at most this many devices (GGML_VK_MAX_DEVICES).
  static constexpr int32_t capacity = 16;
  int32_t status = LLAMA_DART_VULKAN_STATUS_NO_LOADER;
  int32_t count = 0;
  llama_dart_vulkan_device_info devices[capacity] = {};
};

struct llama_dart_vulkan_candidate {
  VkPhysicalDeviceProperties properties = {};
  uint32_t driver_id = 0;
  uint8_t uuid[VK_UUID_SIZE] = {};
  uint8_t luid[VK_LUID_SIZE] = {};
  bool luid_valid = false;
  bool storage_16bit = false;
  uint32_t subgroup_size = 0;
};

// Smaller is preferred. Mirrors the driver priorities of
// ggml_vk_instance_init for two drivers of one GPU.
static inline int llama_dart_vulkan_driver_priority(uint32_t vendor_id,
                                                    uint32_t driver_id) {
  if (driver_id == VK_DRIVER_ID_MESA_DOZEN) {
    return 100;
  }
  switch (vendor_id) {
  case 0x1002: // AMD
    switch (driver_id) {
    case VK_DRIVER_ID_MESA_RADV:
      return 1;
    case VK_DRIVER_ID_AMD_OPEN_SOURCE:
      return 2;
    case VK_DRIVER_ID_AMD_PROPRIETARY:
      return 3;
    }
    break;
  case 0x8086: // Intel
    switch (driver_id) {
    case VK_DRIVER_ID_INTEL_OPEN_SOURCE_MESA:
      return 1;
    case VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS:
      return 2;
    }
    break;
  case 0x10de: // NVIDIA
    switch (driver_id) {
    case VK_DRIVER_ID_NVIDIA_PROPRIETARY:
      return 1;
    case VK_DRIVER_ID_MESA_NVK:
      return 2;
    }
    break;
  case 0x5143: // Qualcomm
    switch (driver_id) {
    case VK_DRIVER_ID_QUALCOMM_PROPRIETARY:
      return 1;
    case VK_DRIVER_ID_MESA_TURNIP:
      return 2;
    }
    break;
  }
  return INT32_MAX;
}

// Mirrors the device selection of ggml_vk_instance_init at llama.cpp v0.6.0:
// the indices of GGML_VK_VISIBLE_DEVICES as given, or else every discrete and
// integrated GPU with 16-bit storage buffers, one per physical GPU, or else
// the first device that is not a CPU.
static inline std::vector<uint32_t> llama_dart_vulkan_select_devices(
    const std::vector<llama_dart_vulkan_candidate> &candidates,
    const char *visible_devices) {
  std::vector<uint32_t> selected;
  if (visible_devices != nullptr) {
    // Read as ggml reads it, with the same standard library calls, so that
    // white space, signs, and numbers that do not fit are treated alike: a
    // value that cannot be read ends the list, and ggml registers no device
    // at all when an index is out of range.
    std::string indices(visible_devices);
    std::replace(indices.begin(), indices.end(), ',', ' ');
    std::stringstream stream(indices);
    size_t index;
    while (stream >> index) {
      if (index >= candidates.size()) {
        return {};
      }
      selected.push_back(static_cast<uint32_t>(index));
    }
    return selected;
  }

  for (uint32_t i = 0; i < candidates.size(); ++i) {
    const llama_dart_vulkan_candidate &candidate = candidates[i];
    const VkPhysicalDeviceType type = candidate.properties.deviceType;
    if ((type != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
         type != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
        !candidate.storage_16bit) {
      continue;
    }
    const auto same_gpu = [&candidates, &candidate](uint32_t k) {
      const llama_dart_vulkan_candidate &old = candidates[k];
      const bool same_id =
          memcmp(old.uuid, candidate.uuid, VK_UUID_SIZE) == 0 ||
          (old.luid_valid && candidate.luid_valid &&
           memcmp(old.luid, candidate.luid, VK_LUID_SIZE) == 0);
      const bool both_moltenvk = old.driver_id == VK_DRIVER_ID_MOLTENVK &&
                                 candidate.driver_id == VK_DRIVER_ID_MOLTENVK;
      return same_id && !both_moltenvk;
    };
    const auto old = std::find_if(selected.begin(), selected.end(), same_gpu);
    if (old == selected.end()) {
      selected.push_back(i);
      continue;
    }
    const llama_dart_vulkan_candidate &previous = candidates[*old];
    const uint32_t vendor_id = previous.properties.vendorID;
    if (llama_dart_vulkan_driver_priority(vendor_id, candidate.driver_id) <
        llama_dart_vulkan_driver_priority(vendor_id, previous.driver_id)) {
      selected.erase(old);
      selected.push_back(i);
    }
  }

  if (selected.empty()) {
    for (uint32_t i = 0; i < candidates.size(); ++i) {
      if (candidates[i].properties.deviceType !=
          VK_PHYSICAL_DEVICE_TYPE_CPU) {
        selected.push_back(i);
        break;
      }
    }
  }
  return selected;
}

// Reads the device facts through the loader entry point
// get_instance_proc_addr. It creates an instance the way ggml-vulkan does and
// destroys it again, and creates no logical device. Every entry point is
// resolved through the loader and checked before use: a 1.0 loader has no
// vkEnumerateInstanceVersion and no vkGetPhysicalDeviceProperties2.
static inline void llama_dart_vulkan_probe(
    PFN_vkGetInstanceProcAddr get_instance_proc_addr,
    const char *visible_devices, llama_dart_vulkan_devices &result) {
  result = llama_dart_vulkan_devices();
  if (get_instance_proc_addr == nullptr) {
    return;
  }
  const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
      get_instance_proc_addr(VK_NULL_HANDLE, "vkCreateInstance"));
  if (create_instance == nullptr) {
    return;
  }
  uint32_t instance_version = VK_API_VERSION_1_0;
  const auto enumerate_instance_version =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(get_instance_proc_addr(
          VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerate_instance_version != nullptr &&
      enumerate_instance_version(&instance_version) != VK_SUCCESS) {
    result.status = LLAMA_DART_VULKAN_STATUS_LOADER_ERROR;
    return;
  }

  // The application ggml-vulkan presents, so that a driver or layer that
  // decides by application sees the same one.
  VkApplicationInfo application = {};
  application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application.pApplicationName = "ggml-vulkan";
  application.applicationVersion = 1;
  application.apiVersion = instance_version;
  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &application;
  VkInstance instance = VK_NULL_HANDLE;
  if (create_instance(&create_info, nullptr, &instance) != VK_SUCCESS ||
      instance == VK_NULL_HANDLE) {
    result.status = LLAMA_DART_VULKAN_STATUS_LOADER_ERROR;
    return;
  }

  const auto destroy_instance = reinterpret_cast<PFN_vkDestroyInstance>(
      get_instance_proc_addr(instance, "vkDestroyInstance"));
  const auto enumerate_devices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          get_instance_proc_addr(instance, "vkEnumeratePhysicalDevices"));
  const auto get_properties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
          get_instance_proc_addr(instance, "vkGetPhysicalDeviceProperties"));
  // Core since 1.1, for an instance and for a physical device alike.
  const bool has_version_1_1 = instance_version >= VK_API_VERSION_1_1;
  const auto get_properties2 =
      !has_version_1_1
          ? nullptr
          : reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                get_instance_proc_addr(instance,
                                       "vkGetPhysicalDeviceProperties2"));
  const auto get_features2 =
      !has_version_1_1
          ? nullptr
          : reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                get_instance_proc_addr(instance,
                                       "vkGetPhysicalDeviceFeatures2"));

  std::vector<VkPhysicalDevice> physical_devices;
  result.status = LLAMA_DART_VULKAN_STATUS_NO_LOADER;
  if (enumerate_devices != nullptr && get_properties != nullptr) {
    uint32_t device_count = 0;
    VkResult enumerated = enumerate_devices(instance, &device_count, nullptr);
    if (enumerated == VK_SUCCESS && device_count > 0) {
      physical_devices.resize(device_count);
      enumerated =
          enumerate_devices(instance, &device_count, physical_devices.data());
      physical_devices.resize(
          std::min<size_t>(device_count, physical_devices.size()));
    }
    result.status = enumerated == VK_SUCCESS || enumerated == VK_INCOMPLETE
                        ? LLAMA_DART_VULKAN_STATUS_OK
                        : LLAMA_DART_VULKAN_STATUS_LOADER_ERROR;
  }

  if (result.status == LLAMA_DART_VULKAN_STATUS_OK) {
    std::vector<llama_dart_vulkan_candidate> candidates(
        physical_devices.size());
    for (size_t i = 0; i < physical_devices.size(); ++i) {
      llama_dart_vulkan_candidate &candidate = candidates[i];
      get_properties(physical_devices[i], &candidate.properties);
      if (get_properties2 == nullptr || get_features2 == nullptr ||
          candidate.properties.apiVersion < VK_API_VERSION_1_1) {
        continue;
      }
      // A driver leaves a structure it does not know untouched, as a 1.1
      // driver does with the driver properties of 1.2.
      VkPhysicalDeviceDriverProperties driver = {};
      driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
      VkPhysicalDeviceIDProperties id = {};
      id.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
      id.pNext = &driver;
      VkPhysicalDeviceSubgroupProperties subgroup = {};
      subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
      subgroup.pNext = &id;
      VkPhysicalDeviceProperties2 properties2 = {};
      properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
      properties2.pNext = &subgroup;
      get_properties2(physical_devices[i], &properties2);
      candidate.driver_id = static_cast<uint32_t>(driver.driverID);
      memcpy(candidate.uuid, id.deviceUUID, VK_UUID_SIZE);
      memcpy(candidate.luid, id.deviceLUID, VK_LUID_SIZE);
      candidate.luid_valid = id.deviceLUIDValid != VK_FALSE;
      candidate.subgroup_size = subgroup.subgroupSize;

      // ggml reads this feature from VkPhysicalDeviceVulkan11Features, which
      // has the same value and which a 1.1 driver does not fill in.
      VkPhysicalDevice16BitStorageFeatures storage = {};
      storage.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
      VkPhysicalDeviceFeatures2 features2 = {};
      features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
      features2.pNext = &storage;
      get_features2(physical_devices[i], &features2);
      candidate.storage_16bit = storage.storageBuffer16BitAccess != VK_FALSE;
    }

    for (const uint32_t index :
         llama_dart_vulkan_select_devices(candidates, visible_devices)) {
      if (result.count == llama_dart_vulkan_devices::capacity) {
        break;
      }
      const llama_dart_vulkan_candidate &candidate = candidates[index];
      llama_dart_vulkan_device_info &info = result.devices[result.count++];
      info.struct_size = sizeof(info);
      info.instance_api_version = instance_version;
      info.physical_device_index = index;
      info.api_version = candidate.properties.apiVersion;
      info.driver_version = candidate.properties.driverVersion;
      info.vendor_id = candidate.properties.vendorID;
      info.device_id = candidate.properties.deviceID;
      info.device_type = static_cast<uint32_t>(candidate.properties.deviceType);
      info.subgroup_size = candidate.subgroup_size;
      static_assert(sizeof(info.device_name) >=
                        sizeof(candidate.properties.deviceName),
                    "device_name holds a Vulkan device name");
      memcpy(info.device_name, candidate.properties.deviceName,
             sizeof(candidate.properties.deviceName));
      info.device_name[sizeof(info.device_name) - 1] = '\0';
    }
  }

  if (destroy_instance != nullptr) {
    destroy_instance(instance, nullptr);
  }
}
