#pragma once

#include "llama_dart_device_memory_internal.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

// System-wide free memory of a GPU device that counts other processes. Only a
// Windows Vulkan or CUDA GPU has one; every other device is not_applicable.
llama_dart_system_memory llama_dart_read_system_memory(ggml_backend_dev_t dev);

// The free memory of a device: the driver's number, lowered to the system's
// on Windows. Unreadable system memory throws when `strict`, else it leaves
// the driver's number.
size_t llama_dart_device_free_bytes(ggml_backend_dev_t dev, size_t driver_free,
                                    bool strict);

// Adds the Windows correction to `margins`, indexed like the model's device
// list for `mparams`. Does nothing elsewhere. Returns true when the model's
// selected devices include a WDDM CUDA GPU.
struct llama_model_params;
bool llama_dart_fit_apply_system_margins(const char *path_model,
                                         const llama_model_params &mparams,
                                         size_t *margins, size_t n_margins);

// Reads available Windows system commit bytes: (CommitLimit - CommitTotal) *
// PageSize. Throws std::runtime_error on OS query failure, zero page size or
// limit, or 64-bit overflow. Not used on non-Windows hosts.
uint64_t llama_dart_read_system_commit_available_bytes();

#if defined(LLAMADART_TESTING) && defined(_WIN32)
bool llama_dart_test_win_dev_override_memory(ggml_backend_dev_t dev,
                                             size_t *free, size_t *total);
bool llama_dart_test_win_dev_override_props(
    ggml_backend_dev_t dev, struct ggml_backend_dev_props *props);
#endif

// The name and LUID of the Vulkan device `index` of the wrapper's probe.
bool llama_dart_vulkan_device_luid(int32_t index, char *name, size_t name_size,
                                   uint8_t *luid, bool *luid_valid);
