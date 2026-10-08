#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_vulkan_internal.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// A Vulkan loader that answers from a table, so that the probe is tested
// against loaders and devices that no test machine has: a 1.0 loader, a
// device below Vulkan 1.2, a subgroup size of 16, and several drivers for one
// GPU.

namespace {

struct fake_device {
  const char *name;
  VkPhysicalDeviceType type;
  uint32_t api_version;
  uint32_t vendor_id;
  uint32_t device_id;
  uint32_t subgroup_size;
  bool storage_16bit;
  VkDriverId driver_id;
  uint8_t uuid;
};

struct fake_loader {
  // 0 for a loader without vkEnumerateInstanceVersion.
  uint32_t instance_version = VK_API_VERSION_1_3;
  std::vector<fake_device> devices;
  std::string missing;
  VkResult create_result = VK_SUCCESS;
  uint32_t requested_api_version = 0;
  std::string application;
  int live_instances = 0;
  int destroyed_instances = 0;
  int properties2_calls = 0;
};

fake_loader g_loader;

VkInstance fake_instance() {
  return reinterpret_cast<VkInstance>(static_cast<uintptr_t>(0x1000));
}

const fake_device &device_of(VkPhysicalDevice handle) {
  return g_loader.devices[reinterpret_cast<uintptr_t>(handle) - 1];
}

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_instance_version(uint32_t *out) {
  *out = g_loader.instance_version;
  return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
fake_create_instance(const VkInstanceCreateInfo *info,
                     const VkAllocationCallbacks *, VkInstance *out) {
  g_loader.requested_api_version = info->pApplicationInfo->apiVersion;
  g_loader.application = info->pApplicationInfo->pApplicationName;
  assert(info->enabledExtensionCount == 0 && info->enabledLayerCount == 0);
  if (g_loader.create_result != VK_SUCCESS) {
    return g_loader.create_result;
  }
  ++g_loader.live_instances;
  *out = fake_instance();
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_destroy_instance(VkInstance instance,
                                                 const VkAllocationCallbacks *) {
  assert(instance == fake_instance());
  --g_loader.live_instances;
  ++g_loader.destroyed_instances;
}

VKAPI_ATTR VkResult VKAPI_CALL fake_enumerate_devices(
    VkInstance instance, uint32_t *count, VkPhysicalDevice *devices) {
  assert(instance == fake_instance());
  const uint32_t available = static_cast<uint32_t>(g_loader.devices.size());
  if (devices == nullptr) {
    *count = available;
    return VK_SUCCESS;
  }
  assert(*count == available);
  for (uint32_t i = 0; i < available; ++i) {
    devices[i] =
        reinterpret_cast<VkPhysicalDevice>(static_cast<uintptr_t>(i + 1));
  }
  return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL fake_get_properties(
    VkPhysicalDevice handle, VkPhysicalDeviceProperties *properties) {
  const fake_device &device = device_of(handle);
  *properties = {};
  properties->apiVersion = device.api_version;
  properties->driverVersion = 7;
  properties->vendorID = device.vendor_id;
  properties->deviceID = device.device_id;
  properties->deviceType = device.type;
  snprintf(properties->deviceName, sizeof(properties->deviceName), "%s",
           device.name);
}

// Fills in only what a driver of the device's version knows, and fails the
// test for a call that the device's version does not have.
VKAPI_ATTR void VKAPI_CALL fake_get_properties2(
    VkPhysicalDevice handle, VkPhysicalDeviceProperties2 *properties) {
  const fake_device &device = device_of(handle);
  assert(g_loader.instance_version >= VK_API_VERSION_1_1);
  assert(device.api_version >= VK_API_VERSION_1_1);
  ++g_loader.properties2_calls;
  for (auto *next = static_cast<VkBaseOutStructure *>(properties->pNext);
       next != nullptr; next = next->pNext) {
    if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES) {
      reinterpret_cast<VkPhysicalDeviceSubgroupProperties *>(next)
          ->subgroupSize = device.subgroup_size;
    } else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES) {
      memset(reinterpret_cast<VkPhysicalDeviceIDProperties *>(next)->deviceUUID,
             device.uuid, VK_UUID_SIZE);
    } else if (next->sType ==
                   VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES &&
               device.api_version >= VK_API_VERSION_1_2) {
      reinterpret_cast<VkPhysicalDeviceDriverProperties *>(next)->driverID =
          device.driver_id;
    }
  }
}

VKAPI_ATTR void VKAPI_CALL fake_get_features2(
    VkPhysicalDevice handle, VkPhysicalDeviceFeatures2 *features) {
  const fake_device &device = device_of(handle);
  assert(device.api_version >= VK_API_VERSION_1_1);
  for (auto *next = static_cast<VkBaseOutStructure *>(features->pNext);
       next != nullptr; next = next->pNext) {
    if (next->sType ==
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES) {
      reinterpret_cast<VkPhysicalDevice16BitStorageFeatures *>(next)
          ->storageBuffer16BitAccess = device.storage_16bit;
    }
  }
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
fake_get_instance_proc_addr(VkInstance instance, const char *name) {
  const std::string entry = name;
  if (entry == g_loader.missing) {
    return nullptr;
  }
  const auto as_void = [](auto function) {
    return reinterpret_cast<PFN_vkVoidFunction>(function);
  };
  if (instance == VK_NULL_HANDLE) {
    if (entry == "vkCreateInstance") {
      return as_void(fake_create_instance);
    }
    if (entry == "vkEnumerateInstanceVersion") {
      return g_loader.instance_version == 0
                 ? nullptr
                 : as_void(fake_enumerate_instance_version);
    }
    return nullptr;
  }
  assert(instance == fake_instance());
  const bool has_version_1_1 = g_loader.instance_version >= VK_API_VERSION_1_1;
  if (entry == "vkDestroyInstance") {
    return as_void(fake_destroy_instance);
  }
  if (entry == "vkEnumeratePhysicalDevices") {
    return as_void(fake_enumerate_devices);
  }
  if (entry == "vkGetPhysicalDeviceProperties") {
    return as_void(fake_get_properties);
  }
  if (entry == "vkGetPhysicalDeviceProperties2" && has_version_1_1) {
    return as_void(fake_get_properties2);
  }
  if (entry == "vkGetPhysicalDeviceFeatures2" && has_version_1_1) {
    return as_void(fake_get_features2);
  }
  return nullptr;
}

const uint32_t kAmd = 0x1002;
const uint32_t kArm = 0x13b5;
const uint32_t kMesa = 0x10005;

fake_device gpu(const char *name, VkPhysicalDeviceType type, uint8_t uuid,
                VkDriverId driver = VK_DRIVER_ID_MESA_RADV,
                uint32_t vendor = kAmd) {
  return {name, type, VK_API_VERSION_1_3, vendor, 0x1234, 64, true, driver,
          uuid};
}

fake_device lavapipe() {
  return {"llvmpipe", VK_PHYSICAL_DEVICE_TYPE_CPU, VK_API_VERSION_1_3, kMesa,
          0, 8, true, VK_DRIVER_ID_MESA_LLVMPIPE, 9};
}

llama_dart_vulkan_devices probe(const std::vector<fake_device> &devices,
                                const char *visible_devices = nullptr) {
  g_loader.devices = devices;
  g_loader.live_instances = 0;
  g_loader.destroyed_instances = 0;
  g_loader.properties2_calls = 0;
  llama_dart_vulkan_devices result;
  llama_dart_vulkan_probe(fake_get_instance_proc_addr, visible_devices,
                          result);
  // No path leaves an instance behind.
  assert(g_loader.live_instances == 0);
  return result;
}

std::vector<std::string> names(const llama_dart_vulkan_devices &result) {
  assert(result.status == LLAMA_DART_VULKAN_STATUS_OK);
  std::vector<std::string> selected;
  for (int32_t i = 0; i < result.count; ++i) {
    selected.push_back(result.devices[i].device_name);
  }
  return selected;
}

using list = std::vector<std::string>;

// A phone GPU whose subgroup size is 16, as the Mali-G715 reports it.
void test_subgroup_size() {
  g_loader = fake_loader();
  const auto result =
      probe({{"Mali-G715", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
              VK_MAKE_API_VERSION(0, 1, 3, 278), kArm, 0xb8a20000, 16, true,
              VK_DRIVER_ID_ARM_PROPRIETARY, 1}});
  assert(result.status == LLAMA_DART_VULKAN_STATUS_OK && result.count == 1);
  const llama_dart_vulkan_device_info &info = result.devices[0];
  assert(info.struct_size == sizeof(info));
  assert(info.instance_api_version == VK_API_VERSION_1_3);
  assert(info.physical_device_index == 0);
  assert(info.api_version == VK_MAKE_API_VERSION(0, 1, 3, 278));
  assert(info.driver_version == 7);
  assert(info.vendor_id == kArm && info.device_id == 0xb8a20000);
  assert(info.device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
  assert(info.subgroup_size == 16);
  assert(std::string(info.device_name) == "Mali-G715");
  // ggml-vulkan's application and API version, and one instance, destroyed.
  assert(g_loader.application == "ggml-vulkan");
  assert(g_loader.requested_api_version == VK_API_VERSION_1_3);
  assert(g_loader.destroyed_instances == 1);
}

// A current loader over a driver that stops at Vulkan 1.1: ggml registers
// the device and then calls 1.2 functions it does not have. The record shows
// the version, and the 1.1 properties are still read.
void test_device_below_1_2() {
  g_loader = fake_loader();
  const auto result =
      probe({{"Mali-G68", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
              VK_MAKE_API_VERSION(0, 1, 1, 177), kArm, 1, 4, true,
              VK_DRIVER_ID_ARM_PROPRIETARY, 1}});
  assert(result.count == 1);
  assert(result.devices[0].instance_api_version == VK_API_VERSION_1_3);
  assert(result.devices[0].api_version == VK_MAKE_API_VERSION(0, 1, 1, 177));
  assert(result.devices[0].subgroup_size == 4);
}

// A 1.1 loader, on which ggml registers no device: the device is listed with
// the loader's version, so a caller can say what is missing.
void test_loader_1_1() {
  g_loader = fake_loader();
  g_loader.instance_version = VK_API_VERSION_1_1;
  const auto result =
      probe({gpu("gpu", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 1)});
  assert(result.count == 1);
  assert(result.devices[0].instance_api_version == VK_API_VERSION_1_1);
  assert(result.devices[0].subgroup_size == 64);
  assert(g_loader.requested_api_version == VK_API_VERSION_1_1);
}

// A 1.0 loader has neither vkEnumerateInstanceVersion nor the 1.1 property
// queries. Nothing that is missing is called, and the subgroup size is
// unknown.
void test_loader_1_0() {
  g_loader = fake_loader();
  g_loader.instance_version = 0;
  const auto result =
      probe({{"old gpu", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
              VK_API_VERSION_1_0, kArm, 1, 4, true,
              VK_DRIVER_ID_ARM_PROPRIETARY, 1}});
  assert(names(result) == list({"old gpu"}));
  assert(result.devices[0].instance_api_version == VK_API_VERSION_1_0);
  assert(result.devices[0].api_version == VK_API_VERSION_1_0);
  assert(result.devices[0].subgroup_size == 0);
  assert(g_loader.requested_api_version == VK_API_VERSION_1_0);
  assert(g_loader.properties2_calls == 0);
}

// A 1.0 device under a current loader is not asked for 1.1 properties.
void test_device_1_0() {
  g_loader = fake_loader();
  const auto result =
      probe({{"old gpu", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
              VK_API_VERSION_1_0, kAmd, 1, 4, true, VK_DRIVER_ID_MESA_RADV,
              1}});
  assert(names(result) == list({"old gpu"}));
  assert(result.devices[0].subgroup_size == 0);
  assert(g_loader.properties2_calls == 0);
}

void test_unavailable() {
  llama_dart_vulkan_devices result;
  llama_dart_vulkan_probe(nullptr, nullptr, result);
  assert(result.status == LLAMA_DART_VULKAN_STATUS_NO_LOADER);
  assert(result.count == 0);

  for (const char *missing :
       {"vkCreateInstance", "vkEnumeratePhysicalDevices",
        "vkGetPhysicalDeviceProperties"}) {
    g_loader = fake_loader();
    g_loader.missing = missing;
    result = probe({gpu("gpu", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1)});
    assert(result.status == LLAMA_DART_VULKAN_STATUS_NO_LOADER);
    assert(result.count == 0);
  }

  // Without the 1.1 queries the device is still listed.
  g_loader = fake_loader();
  g_loader.missing = "vkGetPhysicalDeviceProperties2";
  result = probe({gpu("gpu", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1)});
  assert(names(result) == list({"gpu"}));
  assert(result.devices[0].subgroup_size == 0);

  // No driver: the loader cannot create an instance.
  g_loader = fake_loader();
  g_loader.create_result = VK_ERROR_INCOMPATIBLE_DRIVER;
  result = probe({gpu("gpu", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1)});
  assert(result.status == LLAMA_DART_VULKAN_STATUS_LOADER_ERROR);
  assert(g_loader.destroyed_instances == 0);

  g_loader = fake_loader();
  assert(names(probe({})).empty());
}

// The order of ggml_vk_instance_init: GPUs with 16-bit storage in loader
// order, one entry per physical GPU with the preferred driver, which moves to
// the end when it replaces another.
void test_selection() {
  g_loader = fake_loader();
  fake_device no_storage =
      gpu("no 16-bit storage", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 2);
  no_storage.storage_16bit = false;
  const fake_device integrated =
      gpu("integrated", VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, 3);
  const fake_device radv =
      gpu("radv", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 4);
  const fake_device amdvlk =
      gpu("amdvlk", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 4,
          VK_DRIVER_ID_AMD_OPEN_SOURCE);
  const fake_device virtual_gpu =
      gpu("virtual", VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, 5);

  auto result = probe({lavapipe(), no_storage, integrated, radv, amdvlk});
  assert(names(result) == list({"integrated", "radv"}));
  assert(result.devices[0].physical_device_index == 2);
  assert(result.devices[1].physical_device_index == 3);

  assert(names(probe({amdvlk, integrated, radv})) ==
         list({"integrated", "radv"}));
  assert(names(probe({radv, integrated, amdvlk})) ==
         list({"radv", "integrated"}));

  // Without such a GPU, the first device that is not a CPU.
  assert(names(probe({lavapipe(), no_storage, virtual_gpu})) ==
         list({"no 16-bit storage"}));
  assert(names(probe({lavapipe(), virtual_gpu})) == list({"virtual"}));
  // ggml uses no CPU implementation unless it is asked to.
  assert(names(probe({lavapipe()})).empty());
}

void test_visible_devices() {
  g_loader = fake_loader();
  const std::vector<fake_device> devices = {
      lavapipe(), gpu("a", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 1),
      gpu("b", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, 2)};
  assert(names(probe(devices)) == list({"a", "b"}));
  assert(names(probe(devices, "0")) == list({"llvmpipe"}));
  assert(names(probe(devices, "2,0")) == list({"b", "llvmpipe"}));
  assert(names(probe(devices, "2 1")) == list({"b", "a"}));
  assert(names(probe(devices, "1,x,2")) == list({"a"}));
  // An index out of range leaves ggml without devices, and so does an empty
  // value.
  assert(names(probe(devices, "1,3")).empty());
  assert(names(probe(devices, "-1")).empty());
  assert(names(probe(devices, "")).empty());
}

void test_capacity() {
  g_loader = fake_loader();
  std::vector<fake_device> devices;
  for (uint8_t i = 0; i < 20; ++i) {
    devices.push_back(gpu("gpu", VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, i));
  }
  assert(probe(devices).count == llama_dart_vulkan_devices::capacity);
}

bool is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

// The exports against whatever this machine has. They must answer without a
// Vulkan loader, and LLAMADART_TEST_VULKAN_DEVICES names how many devices a
// machine that is known to have one must report.
void test_exports() {
  llama_dart_vulkan_device_info info = {};
  assert(llama_dart_vulkan_get_device_info(0, nullptr) ==
         LLAMA_DART_VULKAN_STATUS_INVALID_ARGUMENT);
  info.struct_size = sizeof(info) - 1;
  assert(llama_dart_vulkan_get_device_info(0, &info) ==
         LLAMA_DART_VULKAN_STATUS_INVALID_ARGUMENT);
  info.struct_size = sizeof(info);

  const int32_t count = llama_dart_vulkan_get_device_count();
  assert(count == llama_dart_vulkan_get_device_count());
  const char *expected = getenv("LLAMADART_TEST_VULKAN_DEVICES");
#if defined(__APPLE__)
  assert(count == LLAMA_DART_VULKAN_STATUS_UNSUPPORTED);
#endif
  if (count < 0) {
    assert(count == LLAMA_DART_VULKAN_STATUS_UNSUPPORTED ||
           count == LLAMA_DART_VULKAN_STATUS_NO_LOADER ||
           count == LLAMA_DART_VULKAN_STATUS_LOADER_ERROR);
    assert(llama_dart_vulkan_get_device_info(0, &info) == count);
    assert(info.struct_size == sizeof(info) && info.device_name[0] == '\0');
    fprintf(stderr, "Vulkan device facts unavailable: status %d\n", count);
    assert(expected == nullptr);
    return;
  }
  assert(expected == nullptr || count == atoi(expected));
  assert(count <= llama_dart_vulkan_devices::capacity);
  assert(llama_dart_vulkan_get_device_info(-1, &info) ==
         LLAMA_DART_VULKAN_STATUS_NO_DEVICE);
  assert(llama_dart_vulkan_get_device_info(count, &info) ==
         LLAMA_DART_VULKAN_STATUS_NO_DEVICE);
  for (int32_t i = 0; i < count; ++i) {
    info = {};
    info.struct_size = sizeof(info);
    assert(llama_dart_vulkan_get_device_info(i, &info) ==
           LLAMA_DART_VULKAN_STATUS_OK);
    fprintf(stderr,
            "Vulkan device %d: %s, API %u.%u.%u (loader %u.%u), vendor 0x%x, "
            "device 0x%x, type %u, subgroup size %u\n",
            i, info.device_name, VK_API_VERSION_MAJOR(info.api_version),
            VK_API_VERSION_MINOR(info.api_version),
            VK_API_VERSION_PATCH(info.api_version),
            VK_API_VERSION_MAJOR(info.instance_api_version),
            VK_API_VERSION_MINOR(info.instance_api_version), info.vendor_id,
            info.device_id, info.device_type, info.subgroup_size);
    assert(info.struct_size == sizeof(info));
    assert(info.device_name[0] != '\0');
    assert(info.api_version >= VK_API_VERSION_1_0);
    assert(info.instance_api_version >= VK_API_VERSION_1_0);
    if (info.api_version >= VK_API_VERSION_1_1 &&
        info.instance_api_version >= VK_API_VERSION_1_1) {
      assert(is_power_of_two(info.subgroup_size));
    }
  }
}

} // namespace

int main() {
  test_subgroup_size();
  test_device_below_1_2();
  test_loader_1_1();
  test_loader_1_0();
  test_device_1_0();
  test_unavailable();
  test_selection();
  test_visible_devices();
  test_capacity();
  test_exports();
  return 0;
}
