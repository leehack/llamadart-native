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
#if defined(_WIN32)
  test_real_adapters();
#endif
  return 0;
}
