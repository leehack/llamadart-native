#pragma once

#ifndef _WIN32
#error "D3DKMT is Windows only"
#endif

#include "llama_dart_device_memory_internal.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// D3DKMTQueryStatistics is declared in d3dkmthk.h, which has the layout below
// and asserts sizeof(D3DKMT_QUERYSTATISTICS) == 0x328. The call is "reserved
// for system use", so it is resolved from gdi32.dll at run time, and the
// buffer is laid out by hand so that no toolchain needs the header.
// The offsets are those of the 64-bit Windows structure.
static_assert(sizeof(void *) == 8, "the D3DKMT offsets are 64-bit");

namespace llama_dart_d3dkmt {

constexpr size_t kSize = 0x328;
constexpr size_t kTypeOffset = 0;
constexpr size_t kLuidOffset = 4;
constexpr size_t kResultOffset = 24;
constexpr size_t kSegmentIdOffset = 800;
constexpr uint32_t kTypeAdapter = 0;
constexpr uint32_t kTypeSegment = 3;
constexpr size_t kNbSegmentsOffset = kResultOffset + 0;
constexpr size_t kCommitLimitOffset = kResultOffset + 0;
constexpr size_t kBytesResidentOffset = kResultOffset + 16;
constexpr size_t kApertureOffset = kResultOffset + 40;
// Real adapters have a handful of segments.
constexpr uint32_t kMaxSegments = 64;

using QueryStatistics = LONG(WINAPI *)(void *);

inline QueryStatistics resolve() {
  static const QueryStatistics function = []() -> QueryStatistics {
    const HMODULE gdi = LoadLibraryExW(L"gdi32.dll", nullptr,
                                       LOAD_LIBRARY_SEARCH_SYSTEM32);
    return gdi == nullptr ? nullptr
                          : reinterpret_cast<QueryStatistics>(
                                GetProcAddress(gdi, "D3DKMTQueryStatistics"));
  }();
  return function;
}

template <typename T> inline T read(const unsigned char *buffer, size_t offset) {
  T value;
  memcpy(&value, buffer + offset, sizeof(value));
  return value;
}

inline std::string status_text(const char *stage, LONG status) {
  char text[96];
  snprintf(text, sizeof(text), "D3DKMTQueryStatistics %s failed with 0x%08lX",
           stage, static_cast<unsigned long>(status));
  return text;
}

// The memory segments of the adapter with this LUID, for every process. On
// failure `error` says what failed and `segments` is empty.
inline bool read_segments(const unsigned char *luid,
                          std::vector<llama_dart_adapter_segment> &segments,
                          std::string &error) {
  segments.clear();
  static const unsigned char zero[8] = {};
  if (memcmp(luid, zero, sizeof(zero)) == 0) {
    error = "the device has no adapter LUID";
    return false;
  }
  const QueryStatistics query = resolve();
  if (query == nullptr) {
    error = "gdi32.dll has no D3DKMTQueryStatistics";
    return false;
  }
  alignas(8) unsigned char buffer[kSize];
  const auto prepare = [&](uint32_t type) {
    memset(buffer, 0, sizeof(buffer));
    memcpy(buffer + kTypeOffset, &type, sizeof(type));
    memcpy(buffer + kLuidOffset, luid, 8);
  };
  prepare(kTypeAdapter);
  LONG status = query(buffer);
  if (status != 0) {
    error = status_text("adapter query", status);
    return false;
  }
  const uint32_t count = read<uint32_t>(buffer, kNbSegmentsOffset);
  if (count > kMaxSegments) {
    error = "implausible segment count " + std::to_string(count);
    return false;
  }
  for (uint32_t id = 0; id < count; ++id) {
    prepare(kTypeSegment);
    memcpy(buffer + kSegmentIdOffset, &id, sizeof(id));
    status = query(buffer);
    if (status != 0) {
      segments.clear();
      error = status_text("segment query", status);
      return false;
    }
    llama_dart_adapter_segment segment;
    segment.commit_limit = read<uint64_t>(buffer, kCommitLimitOffset);
    segment.bytes_resident = read<uint64_t>(buffer, kBytesResidentOffset);
    segment.aperture = read<uint32_t>(buffer, kApertureOffset) != 0;
    segments.push_back(segment);
  }
  return true;
}

} // namespace llama_dart_d3dkmt
