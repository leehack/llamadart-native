#pragma once

#if defined(LLAMADART_TESTING)

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
  const char *name;
  bool is_gpu;
  bool is_host;
  int64_t total_bytes;
  int64_t free_bytes;
  uint64_t model_bytes;
  uint64_t context_bytes;
  uint64_t compute_bytes;
};

struct llama_dart_test_fit_commit_config {
  bool enabled = false;
  bool force_wddm_cuda_selected = false;
  bool override_measurement = false;
  const llama_dart_test_fit_row *rows = nullptr;
  size_t n_rows = 0;
  bool counter_ok = true;
  uint64_t commit_limit_pages = 0;
  uint64_t commit_total_pages = 0;
  uint64_t page_size = 4096;
  const char *counter_error = nullptr;
};

extern "C" {

LLAMADART_API void llama_dart_test_set_fit_commit_config(
    const llama_dart_test_fit_commit_config *config);

LLAMADART_API uint64_t
llama_dart_test_read_system_commit_available_bytes(void);

#if defined(_WIN32)
typedef long(__stdcall *llama_dart_test_kmt_query_fn)(void *);

struct llama_dart_test_win_dev_config {
  bool enabled = false;
  const char *name = nullptr;
  const char *description = nullptr;
  int type = 0;
  size_t memory_free = 0;
  size_t memory_total = 0;
  bool has_cuda_api = false;
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
