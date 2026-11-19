#pragma once

#if defined(LLAMADART_TESTING)

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

#if defined(_WIN32) && !defined(LLAMADART_API)
#define LLAMADART_TEST_HOOK_API __declspec(dllexport)
#else
#define LLAMADART_TEST_HOOK_API
#endif

#if !defined(LLAMADART_API)
#define LLAMADART_API LLAMADART_TEST_HOOK_API
#endif

struct llama_dart_test_fit_row {
  ggml_backend_dev_t device;
  bool is_host;
  int64_t total_bytes;
  int64_t free_bytes;
  uint64_t model_bytes;
  uint64_t context_bytes;
  uint64_t compute_bytes;
};

extern "C" {

LLAMADART_API void llama_dart_test_set_fit_rows(
    const llama_dart_test_fit_row *rows, size_t count);

#if defined(_WIN32)
typedef long(__stdcall *llama_dart_test_kmt_query_fn)(void *);

struct llama_dart_test_win_dev_config {
  ggml_backend_dev_t cuda_device = nullptr;
  ggml_backend_dev_t vulkan_device = nullptr;
  uint8_t vulkan_luid[8] = {};
  int (*cuda_init)(unsigned flags) = nullptr;
  int (*cuda_device_count)(int *count) = nullptr;
  int (*cuda_device_get)(int *device, int ordinal) = nullptr;
  int (*cuda_device_name)(char *name, int length, int device) = nullptr;
  int (*cuda_device_luid)(char *luid, unsigned *node_mask,
                          int device) = nullptr;
  int32_t cuda_registered_count = 0;
  llama_dart_test_kmt_query_fn kmt_query = nullptr;
};

LLAMADART_API void llama_dart_test_set_win_dev_config(
    const llama_dart_test_win_dev_config *config);
#endif
}

#endif
