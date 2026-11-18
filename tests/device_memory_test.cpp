#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_device_memory_internal.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include "llama_dart_cuda_luid_internal.h"
#include "llama_dart_d3dkmt_internal.h"
#include <dxgi.h>
#endif

namespace {

const uint64_t kMiB = 1024ull * 1024ull;

llama_dart_adapter_segment segment(uint64_t limit_mib, uint64_t resident_mib,
                                   bool aperture) {
  llama_dart_adapter_segment result;
  result.commit_limit = limit_mib * kMiB;
  result.bytes_resident = resident_mib * kMiB;
  result.aperture = aperture;
  return result;
}

// The L4 of the recorded runs: 22,595 MiB dedicated, a CUDA process holding
// 16,332 MiB, and an aperture segment that must not count.
void test_discrete_adapter() {
  const std::vector<llama_dart_adapter_segment> segments = {
      segment(22595, 16332, false), segment(4096, 4096, true)};
  uint64_t free_bytes = 0;
  assert(llama_dart_adapter_free_bytes(segments, 22595 * kMiB + kMiB / 2,
                                       false, &free_bytes));
  assert(free_bytes == 6263 * kMiB);
}

void test_integrated_adapter() {
  const std::vector<llama_dart_adapter_segment> segments = {
      segment(1024, 100, false), segment(8192, 2000, true)};
  uint64_t free_bytes = 0;
  assert(llama_dart_adapter_free_bytes(segments, 16384 * kMiB, true,
                                       &free_bytes));
  assert(free_bytes == (1024 + 8192 - 2100) * kMiB);
}

void test_degenerate_adapters() {
  uint64_t free_bytes = 7;
  assert(!llama_dart_adapter_free_bytes({}, 1000 * kMiB, false, &free_bytes));
  assert(!llama_dart_adapter_free_bytes({segment(4096, 4096, true)},
                                        1000 * kMiB, false, &free_bytes));
  assert(free_bytes == 7);

  // Resident above capacity floors at 0.
  assert(llama_dart_adapter_free_bytes({segment(1000, 1200, false)},
                                       1000 * kMiB, false, &free_bytes));
  assert(free_bytes == 0);

  // A commit limit of 0 leaves the driver's total as the capacity.
  assert(llama_dart_adapter_free_bytes({segment(0, 300, false)}, 1000 * kMiB,
                                       false, &free_bytes));
  assert(free_bytes == 700 * kMiB);

  // The driver's total bounds a larger commit limit.
  assert(llama_dart_adapter_free_bytes({segment(5000, 300, false)},
                                       1000 * kMiB, false, &free_bytes));
  assert(free_bytes == 700 * kMiB);
}

void test_clamp() {
  assert(llama_dart_effective_free(21827 * kMiB, 6263 * kMiB) == 6263 * kMiB);
  assert(llama_dart_effective_free(5000 * kMiB, 6263 * kMiB) == 5000 * kMiB);
}

void test_device_index() {
  int32_t index = -1;
  assert(llama_dart_vulkan_device_index("Vulkan0", &index) && index == 0);
  assert(llama_dart_vulkan_device_index("Vulkan12", &index) && index == 12);
  for (const char *name :
       {"Vulkan_Host", "Vulkan", "Vulkan1x", "CUDA0", "vulkan0", "Vulkan-1",
        "Vulkan00000", "", "XVulkan0"}) {
    index = -1;
    assert(!llama_dart_vulkan_device_index(name, &index));
    assert(index == -1);
  }
}


void test_cuda_device_index() {
  int32_t index = -1;
  assert(llama_dart_device_index("CUDA0", "CUDA", &index) && index == 0);
  assert(llama_dart_device_index("CUDA12", "CUDA", &index) && index == 12);
  for (const char *name : {"CUDA", "CUDA_Host", "CUDA1x", "Vulkan0", "cuda0",
                           "CUDA-1", "CUDA00000", "", "XCUDA0"}) {
    index = -1;
    assert(!llama_dart_device_index(name, "CUDA", &index));
    assert(index == -1);
  }
  assert(llama_dart_device_index("Vulkan3", "Vulkan", &index) && index == 3);
  assert(!llama_dart_vulkan_device_index("CUDA0", &index));
}

void test_memory_source() {
  using source = llama_dart_memory_source;
  assert(llama_dart_device_memory_source("CUDA0", true) == source::cuda);
  assert(llama_dart_device_memory_source("CUDA3", true) == source::cuda);
  assert(llama_dart_device_memory_source("Vulkan0", true) == source::vulkan);
  assert(llama_dart_device_memory_source("CUDA0", false) == source::none);
  assert(llama_dart_device_memory_source("Vulkan0", false) == source::none);
  for (const char *name : {"CUDA_Host", "Metal", "BLAS", "CPU", ""}) {
    assert(llama_dart_device_memory_source(name, true) == source::none);
  }
}

// Fake driver calls; the state is global because the table holds plain
// function pointers.
struct fake_cuda {
  int count = 2;
  int count_result = 0;
  int get_result = 0;
  int name_result = 0;
  int luid_result = 0;
  int init_result = 0;
  int init_calls = 0;
  int last_ordinal = -1;
  std::string names[2] = {"NVIDIA L4", "NVIDIA A10"};
  unsigned char luids[2][8] = {{0x5D, 0x50, 0, 0, 0x12, 0x34, 0x56, 0x78},
                               {0x6E, 0x61, 0, 0, 0x9A, 0xBC, 0xDE, 0xF0}};
  int name_ordinal = -1;
  int luid_ordinal = -1;
} g_cuda;

int fake_init(unsigned) {
  ++g_cuda.init_calls;
  return g_cuda.init_result;
}
int fake_count(int *count) {
  *count = g_cuda.count;
  return g_cuda.count_result;
}
int fake_get(int *device, int ordinal) {
  g_cuda.last_ordinal = ordinal;
  *device = ordinal;
  return g_cuda.get_result;
}
int fake_name(char *name, int length, int device) {
  g_cuda.name_ordinal = device;
  snprintf(name, static_cast<size_t>(length), "%s",
           g_cuda.names[device].c_str());
  return g_cuda.name_result;
}
int fake_luid(char *luid, unsigned *mask, int device) {
  g_cuda.luid_ordinal = device;
  memcpy(luid, g_cuda.luids[device], 8);
  *mask = 1;
  return g_cuda.luid_result;
}

llama_dart_cuda_api fake_api() {
  g_cuda = fake_cuda();
  llama_dart_cuda_api api;
  api.init = fake_init;
  api.device_count = fake_count;
  api.device_get = fake_get;
  api.device_name = fake_name;
  api.device_luid = fake_luid;
  return api;
}

bool map_luid(const llama_dart_cuda_api &api, int32_t index, int32_t registered,
              const char *description, uint8_t *luid, std::string *detail) {
  return llama_dart_cuda_device_luid(api, index, registered, description, luid,
                                     detail);
}

void test_cuda_luid_mapping() {
  const llama_dart_cuda_api api = fake_api();
  uint8_t luid[8] = {};
  std::string detail;
  assert(map_luid(api, 1, 2, "NVIDIA A10", luid, &detail));
  assert(g_cuda.last_ordinal == 1 && g_cuda.name_ordinal == 1 &&
         g_cuda.luid_ordinal == 1);
  assert(g_cuda.init_calls == 1);
  assert(memcmp(luid, g_cuda.luids[1], 8) == 0);

  assert(map_luid(api, 0, 2, "NVIDIA L4", luid, &detail));
  assert(memcmp(luid, g_cuda.luids[0], 8) == 0);
}

void expect_mapping_failure(const llama_dart_cuda_api &api, int32_t index,
                            int32_t registered, const char *description,
                            const char *needle) {
  uint8_t luid[8] = {};
  std::string detail;
  assert(!map_luid(api, index, registered, description, luid, &detail));
  assert(detail.find(needle) != std::string::npos);
}

void test_cuda_luid_mapping_failures() {
  for (int field = 0; field < 5; ++field) {
    llama_dart_cuda_api api = fake_api();
    switch (field) {
      case 0: api.init = nullptr; break;
      case 1: api.device_count = nullptr; break;
      case 2: api.device_get = nullptr; break;
      case 3: api.device_name = nullptr; break;
      case 4: api.device_luid = nullptr; break;
    }
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "nvcuda.dll is not loaded");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.init_result = 3;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "cuInit");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.count_result = 1;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "cuDeviceGetCount");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.count = 1;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "GGML_CUDA_DEVICES");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.count = 3;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "GGML_CUDA_DEVICES");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    expect_mapping_failure(api, 2, 2, "NVIDIA L4", "index");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.get_result = 1;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "cuDeviceGet failed with");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.name_result = 1;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "cuDeviceGetName");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    expect_mapping_failure(api, 0, 2, "NVIDIA A10", "different device");
    expect_mapping_failure(api, 0, 2, nullptr, "different device");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    g_cuda.luid_result = 1;
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "cuDeviceGetLuid");
  }
  {
    const llama_dart_cuda_api api = fake_api();
    memset(g_cuda.luids[0], 0, 8);
    expect_mapping_failure(api, 0, 2, "NVIDIA L4", "no adapter LUID");
  }
}

// The L4 under a foreign hog (recorded P0 numbers): the driver's total is
// 23,033.6 MiB, the adapter's commit limit 22,595.5 MiB, 16,337.7 MiB resident.
void test_system_memory_cuda() {
  const unsigned char want_luid[8] = {0x5D, 0x50, 0, 0, 0, 0, 0, 0};
  unsigned char seen_luid[8] = {};
  bool segments_read = false;
  const uint64_t total = 23033 * kMiB + kMiB * 6 / 10;
  const llama_dart_system_memory result = llama_dart_system_memory_from(
      llama_dart_memory_source::cuda, total, false,
      [&](uint8_t *luid, std::string *) {
        memcpy(luid, want_luid, 8);
        return true;
      },
      [&](const uint8_t *luid, std::vector<llama_dart_adapter_segment> *out,
          std::string *) {
        segments_read = true;
        memcpy(seen_luid, luid, 8);
        out->push_back(segment(0, 0, false));
        (*out)[0].commit_limit = 22595 * kMiB + kMiB / 2;
        (*out)[0].bytes_resident = 16337 * kMiB + kMiB * 7 / 10;
        out->push_back(segment(4096, 4096, true));
        return true;
      });
  assert(result.read == llama_dart_system_read::ok);
  assert(segments_read && memcmp(seen_luid, want_luid, 8) == 0);
  assert(result.free_bytes == 22595 * kMiB + kMiB / 2 - (16337 * kMiB + kMiB * 7 / 10));
}

void test_system_memory_segment_failure() {
  const llama_dart_system_memory result = llama_dart_system_memory_from(
      llama_dart_memory_source::cuda, 1000 * kMiB, false,
      [](uint8_t *luid, std::string *) {
        luid[0] = 1;
        return true;
      },
      [](const uint8_t *, std::vector<llama_dart_adapter_segment> *,
         std::string *detail) {
        *detail = "D3DKMTQueryStatistics adapter query failed";
        return false;
      });
  assert(result.read == llama_dart_system_read::unavailable);
  assert(result.detail.find("D3DKMTQueryStatistics") != std::string::npos);
}

void test_system_memory_no_dedicated_segment() {
  const llama_dart_system_memory result = llama_dart_system_memory_from(
      llama_dart_memory_source::cuda, 1000 * kMiB, false,
      [](uint8_t *luid, std::string *) {
        luid[0] = 1;
        return true;
      },
      [](const uint8_t *, std::vector<llama_dart_adapter_segment> *out,
         std::string *) {
        out->push_back(segment(4096, 0, true));
        return true;
      });
  assert(result.read == llama_dart_system_read::unavailable);
}

// The Vulkan branch of #203 through the same seam.
void test_system_memory_vulkan() {
  auto segments_ok = [](const uint8_t *, std::vector<llama_dart_adapter_segment> *out,
                        std::string *) {
    out->push_back(segment(22595, 16332, false));
    return true;
  };
  llama_dart_system_memory result = llama_dart_system_memory_from(
      llama_dart_memory_source::vulkan, 22595 * kMiB, false,
      [](uint8_t *luid, std::string *) {
        luid[0] = 1;
        return true;
      },
      segments_ok);
  assert(result.read == llama_dart_system_read::ok);
  assert(result.free_bytes == 6263 * kMiB);

  for (const char *why : {"the Vulkan probe has no such device",
                          "the Vulkan probe lists a different device at its index",
                          "the Vulkan device has no adapter LUID"}) {
    result = llama_dart_system_memory_from(
        llama_dart_memory_source::vulkan, 22595 * kMiB, false,
        [why](uint8_t *, std::string *detail) {
          *detail = why;
          return false;
        },
        segments_ok);
    assert(result.read == llama_dart_system_read::unavailable);
    assert(result.detail == why);
  }

  result = llama_dart_system_memory_from(
      llama_dart_memory_source::none, 1, false,
      [](uint8_t *, std::string *) { return true; }, segments_ok);
  assert(result.read == llama_dart_system_read::not_applicable);
}

void test_margins_cuda_device() {
  size_t margins[1] = {static_cast<size_t>(1024 * kMiB)};
  const uint64_t driver = 21663 * kMiB + kMiB / 2;
  const uint64_t system = 6091 * kMiB + kMiB * 4 / 10;
  llama_dart_fit_margins(
      {"CUDA0"}, margins, [&](size_t) { return driver; },
      [&](size_t) {
        llama_dart_system_memory result;
        result.read = llama_dart_system_read::ok;
        result.free_bytes = system;
        return result;
      });
  assert(margins[0] == 1024 * kMiB + (driver - system));

  margins[0] = 1024 * kMiB;
  bool threw = false;
  try {
    llama_dart_fit_margins(
        {"CUDA0"}, margins, [&](size_t) { return driver; },
        [](size_t) {
          llama_dart_system_memory result;
          result.read = llama_dart_system_read::unavailable;
          result.detail = "nvcuda.dll is not loaded";
          return result;
        });
  } catch (const std::runtime_error &error) {
    threw = true;
    assert(std::string(error.what()).find("CUDA0") != std::string::npos);
  }
  assert(threw);
}

#if defined(_WIN32)
// The runners have no NVIDIA driver: the real resolver must report that
// instead of faulting.
void test_real_cuda_resolver() {
  const llama_dart_cuda_api api = llama_dart_cuda_resolve_api();
  if (api.device_luid != nullptr) {
    return;
  }
  uint8_t luid[8] = {};
  std::string detail;
  assert(!llama_dart_cuda_device_luid(api, 0, 1, "x", luid, &detail));
  assert(detail.find("nvcuda.dll is not loaded") != std::string::npos);
}
#endif

llama_dart_system_memory system_ok(uint64_t free_mib) {
  llama_dart_system_memory result;
  result.read = llama_dart_system_read::ok;
  result.free_bytes = free_mib * kMiB;
  return result;
}

void test_margins_one_device() {
  size_t margins[1] = {static_cast<size_t>(1024 * kMiB)};
  llama_dart_fit_margins(
      {"Vulkan0"}, margins, [](size_t) { return 21827 * kMiB; },
      [](size_t) { return system_ok(6263); });
  assert(margins[0] == 16588 * kMiB);
}

void test_margins_system_above_driver() {
  size_t margins[1] = {static_cast<size_t>(1024 * kMiB)};
  llama_dart_fit_margins(
      {"Vulkan0"}, margins, [](size_t) { return 5000 * kMiB; },
      [](size_t) { return system_ok(6263); });
  assert(margins[0] == 1024 * kMiB);
}

void test_margins_each_device_has_its_own_delta() {
  size_t margins[3] = {static_cast<size_t>(100 * kMiB),
                       static_cast<size_t>(200 * kMiB),
                       static_cast<size_t>(300 * kMiB)};
  const uint64_t driver[3] = {1000 * kMiB, 2000 * kMiB, 3000 * kMiB};
  llama_dart_fit_margins(
      {"Vulkan0", "Vulkan1", "CUDA0"}, margins,
      [&](size_t i) { return driver[i]; },
      [](size_t i) {
        if (i == 0) {
          return system_ok(400);
        }
        if (i == 1) {
          return system_ok(1500);
        }
        return llama_dart_system_memory();
      });
  assert(margins[0] == (100 + 600) * kMiB);
  assert(margins[1] == (200 + 500) * kMiB);
  assert(margins[2] == 300 * kMiB);
}

void test_margins_unreadable_device_throws() {
  size_t margins[2] = {0, 0};
  bool threw = false;
  try {
    llama_dart_fit_margins(
        {"Vulkan0", "Vulkan1"}, margins, [](size_t) { return kMiB; },
        [](size_t i) {
          if (i == 0) {
            return system_ok(1);
          }
          llama_dart_system_memory result;
          result.read = llama_dart_system_read::unavailable;
          result.detail = "D3DKMTQueryStatistics adapter query failed";
          return result;
        });
  } catch (const std::runtime_error &error) {
    threw = true;
    const std::string message = error.what();
    assert(message.find("Vulkan1") != std::string::npos);
    assert(message.find("D3DKMTQueryStatistics") != std::string::npos);
  }
  assert(threw);
}

#if defined(_WIN32)
// The runners have no GPU, so this checks the call and the buffer layout, not
// the numbers.
void test_real_adapters() {
  const unsigned char zero[8] = {};
  std::vector<llama_dart_adapter_segment> segments;
  std::string error;
  assert(!llama_dart_d3dkmt::read_segments(zero, segments, error));
  assert(segments.empty() && !error.empty());

  IDXGIFactory1 *factory = nullptr;
  if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                reinterpret_cast<void **>(&factory))) ||
      factory == nullptr) {
    return;
  }
  IDXGIAdapter1 *adapter = nullptr;
  for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND;
       ++i) {
    DXGI_ADAPTER_DESC1 description = {};
    adapter->GetDesc1(&description);
    unsigned char luid[8];
    memcpy(luid, &description.AdapterLuid, sizeof(luid));
    segments.clear();
    error.clear();
    if (llama_dart_d3dkmt::read_segments(luid, segments, error)) {
      assert(segments.size() <= llama_dart_d3dkmt::kMaxSegments);
    } else {
      assert(segments.empty() && !error.empty());
    }
    printf("adapter %u: %zu segments%s%s\n", i, segments.size(),
           error.empty() ? "" : ", ", error.c_str());
    adapter->Release();
  }
  factory->Release();
}
#endif

} // namespace

int main() {
  test_discrete_adapter();
  test_integrated_adapter();
  test_degenerate_adapters();
  test_clamp();
  test_device_index();
  test_margins_one_device();
  test_margins_system_above_driver();
  test_margins_each_device_has_its_own_delta();
  test_margins_unreadable_device_throws();
  test_cuda_device_index();
  test_memory_source();
  test_cuda_luid_mapping();
  test_cuda_luid_mapping_failures();
  test_system_memory_cuda();
  test_system_memory_segment_failure();
  test_system_memory_no_dedicated_segment();
  test_system_memory_vulkan();
  test_margins_cuda_device();
#if defined(_WIN32)
  test_real_cuda_resolver();
  test_real_adapters();
#endif
  return 0;
}
