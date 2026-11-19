#include "llama_dart_wrapper.h"
#include "llama_dart_fit_test_hooks.h"
#include "ggml-backend-impl.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef NDEBUG
#error "Wrapper contract tests require active assertions"
#endif

namespace {

const uint32_t kEmbedding = 256;
const uint32_t kFeedForward = 1024;
const uint32_t kBlocks = 8;
const uint32_t kVocabulary = 256;
const uint32_t kTrainedContext = 65536;
const size_t kMiB = 1024 * 1024;

void add_tensor(gguf_context *gguf, ggml_context *ggml, const std::string &name,
                int64_t columns, int64_t rows) {
  ggml_tensor *tensor =
      rows == 0 ? ggml_new_tensor_1d(ggml, GGML_TYPE_F32, columns)
                : ggml_new_tensor_2d(ggml, GGML_TYPE_F32, columns, rows);
  ggml_set_name(tensor, name.c_str());
  static uint32_t seed = 1;
  float *data = static_cast<float *>(tensor->data);
  for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
    seed = seed * 1664525u + 1013904223u;
    data[i] = rows == 0 ? 1.0f
                        : (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.1f;
  }
  gguf_add_tensor(gguf, tensor);
}

// A llama decoder of about 32.5 MiB with a long trained context, so that its
// weights and its cache are large enough to be fitted into a budget of a few
// tens of MiB. With a NextN layer it has one more block, which the loader
// counts but llama_model_n_layer does not, and which has no weights here.
void make_model(const char *path, bool with_nextn = false) {
  const uint32_t blocks = kBlocks + (with_nextn ? 1 : 0);
  gguf_context *gguf = gguf_init_empty();
  const auto set = [&](const char *key, uint32_t value) {
    gguf_set_val_u32(gguf, (std::string("llama.") + key).c_str(), value);
  };
  gguf_set_val_str(gguf, "general.architecture", "llama");
  gguf_set_val_str(gguf, "general.name", "llamadart fit test");
  set("context_length", kTrainedContext);
  set("embedding_length", kEmbedding);
  set("block_count", blocks);
  if (with_nextn) {
    set("nextn_predict_layers", 1);
  }
  set("feed_forward_length", kFeedForward);
  set("attention.head_count", 4);
  set("vocab_size", kVocabulary);
  gguf_set_val_f32(gguf, "llama.attention.layer_norm_rms_epsilon", 1e-5f);
  gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");

  ggml_init_params params{};
  params.mem_size = 96 * kMiB;
  ggml_context *ggml = ggml_init(params);
  assert(ggml != nullptr);
  const auto add = [&](const std::string &name, int64_t columns, int64_t rows) {
    add_tensor(gguf, ggml, name + ".weight", columns, rows);
  };
  add("token_embd", kEmbedding, kVocabulary);
  for (uint32_t block = 0; block < kBlocks; ++block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    add(prefix + "attn_q", kEmbedding, kEmbedding);
    add(prefix + "attn_k", kEmbedding, kEmbedding);
    add(prefix + "attn_v", kEmbedding, kEmbedding);
    add(prefix + "attn_output", kEmbedding, kEmbedding);
    add(prefix + "ffn_up", kEmbedding, kFeedForward);
    add(prefix + "ffn_down", kFeedForward, kEmbedding);
    add(prefix + "ffn_gate", kEmbedding, kFeedForward);
    add(prefix + "attn_norm", kEmbedding, 0);
    add(prefix + "ffn_norm", kEmbedding, 0);
  }
  add("output_norm", kEmbedding, 0);
  add("output", kEmbedding, kVocabulary);
  const bool written = gguf_write_to_file(gguf, path, false);
  assert(written);
  gguf_free(gguf);
  ggml_free(ggml);
}

struct fit_request {
  llama_model_params model = llama_model_default_params();
  llama_context_params context = llama_context_default_params();
  std::vector<float> split = std::vector<float>(llama_max_devices(), 0.0f);
  std::vector<llama_model_tensor_buft_override> overrides =
      std::vector<llama_model_tensor_buft_override>(
          llama_max_tensor_buft_overrides());
  std::vector<size_t> margins = std::vector<size_t>(llama_max_devices(), 0);
  llama_dart_fit_report report{};

  fit_request() { context.n_ctx = 2048; }

  int32_t run(const char *path, uint32_t n_ctx_min = UINT32_MAX) {
    return llama_dart_fit_params(path, &model, &context, split.data(),
                                 overrides.data(), margins.data(), n_ctx_min,
                                 &report);
  }
};

uint64_t projected_bytes(const llama_dart_fit_device &device) {
  return device.model_bytes + device.context_bytes + device.compute_bytes;
}

void set_budget(fit_request &request, const llama_dart_fit_device &device,
                uint64_t budget) {
  assert(device.free_bytes > 0 && static_cast<uint64_t>(device.free_bytes) > budget);
  request.margins.assign(request.margins.size(),
                         static_cast<size_t>(device.free_bytes) - budget);
}

// Whether the device runs an F32 matrix product. A device that does not, such
// as a Metal device without simdgroup reduction, gets no weights from llama.cpp.
bool device_runs_matmul(const char *name) {
  ggml_backend_dev_t device = ggml_backend_dev_by_name(name);
  assert(device != nullptr);
  ggml_init_params params{};
  params.mem_size = 1 * kMiB;
  params.no_alloc = true;
  ggml_context *ggml = ggml_init(params);
  assert(ggml != nullptr);
  ggml_tensor *weights = ggml_new_tensor_2d(ggml, GGML_TYPE_F32, kEmbedding, kEmbedding);
  ggml_tensor *input = ggml_new_tensor_2d(ggml, GGML_TYPE_F32, kEmbedding, 4);
  ggml_tensor *product = ggml_mul_mat(ggml, weights, input);
  const bool supported = ggml_backend_dev_supports_op(device, product);
  ggml_free(ggml);
  return supported;
}

void test_contract(const char *path) {
  assert(llama_dart_fit_api_version() == 1);

  fit_request request;
  float split[1] = {};
  llama_model_tensor_buft_override overrides[1] = {};
  size_t margins[1] = {};
  assert(llama_dart_fit_params(nullptr, &request.model, &request.context, split,
                               overrides, margins, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_last_error() != nullptr);
  assert(llama_dart_fit_params(path, nullptr, &request.context, split,
                               overrides, margins, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_fit_params(path, &request.model, nullptr, split, overrides,
                               margins, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_fit_params(path, &request.model, &request.context, nullptr,
                               overrides, margins, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_fit_params(path, &request.model, &request.context, split,
                               nullptr, margins, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_fit_params(path, &request.model, &request.context, split,
                               overrides, nullptr, UINT32_MAX, nullptr) ==
         LLAMA_DART_STATUS_EXCEPTION);

  const auto missing = request.run("fit_api_test_missing_model.gguf");
  assert(missing == LLAMA_DART_FIT_STATUS_ERROR);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 2048);
}

// The barrier catches what llama.cpp throws on a file that is not a model.
void test_unreadable_model() {
  const char *path = "fit_api_test_unreadable.gguf";
  FILE *file = std::fopen(path, "wb");
  assert(file != nullptr);
  std::fputs("not a gguf file", file);
  std::fclose(file);
  fit_request request;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_ERROR);
  std::remove(path);
}

bool has_gpu(const char *path) {
  fit_request request;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  return request.report.n_devices > 0;
}

void test_cpu_only(const char *path) {
  ggml_backend_dev_t no_gpu[] = {nullptr};
  fit_request request;
  request.model.devices = no_gpu;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.report.n_devices == 0);
  assert(request.report.n_layers == kBlocks);
  assert(request.report.n_ctx_train == kTrainedContext);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 2048);
  assert(std::string(request.report.host.name) == "Host");
  assert(request.report.host.total_bytes > 0 && request.report.host.free_bytes > 0);
}

void test_missing_measured_device(const char *path) {
  const llama_dart_test_fit_row rows[] = {
      {nullptr, false, 16 * kMiB, 8 * kMiB, 0, 0, 0},
      {nullptr, true, 128 * kMiB, 64 * kMiB, 0, 0, 0},
  };
  llama_dart_test_set_fit_rows(rows, 2);

  ggml_backend_dev_t no_gpu[] = {nullptr};
  fit_request request;
  request.model.devices = no_gpu;
  request.context.n_ctx = 0;
  request.report.n_layers = 777;
  const int32_t status = request.run(path);
  llama_dart_test_set_fit_rows(nullptr, 0);

  assert(status == LLAMA_DART_STATUS_EXCEPTION);
  const char *error = llama_dart_last_error();
  assert(error != nullptr &&
         std::string(error).find("missing backend device") != std::string::npos);
  assert(request.model.devices == no_gpu);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 0);
  assert(request.report.n_layers == 777);
}

void test_everything_fits(const char *path) {
  fit_request request;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 2048);
  assert(request.report.n_layers == kBlocks);
  assert(request.report.n_layers_skipped == 0);
  const llama_dart_fit_device &device = request.report.devices[0];
  assert(device.name[0] != '\0');
  assert(device.free_bytes > 0 && device.total_bytes >= device.free_bytes);
  assert(projected_bytes(device) > 0);
  // The weights are on the device only where it can run the products.
  if (device_runs_matmul(device.name)) {
    assert(device.model_bytes >= 30 * kMiB);
  } else {
    assert(request.report.host.model_bytes >= 30 * kMiB);
  }
}

void test_partial_fit(const char *path) {
  fit_request full;
  assert(full.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  const llama_dart_fit_device device = full.report.devices[0];
  const uint64_t budget = projected_bytes(device) * 6 / 10;

  fit_request request;
  set_budget(request, device, budget);
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.model.n_gpu_layers > 0);
  assert(request.model.n_gpu_layers < static_cast<int32_t>(request.report.n_layers) + 1);
  assert(projected_bytes(request.report.devices[0]) <= budget);
  assert(request.model.tensor_split == request.split.data());
  assert(request.context.n_ctx == 2048);
  std::printf("partial fit: n_gpu_layers=%d projected=%llu budget=%llu\n",
              request.model.n_gpu_layers,
              static_cast<unsigned long long>(projected_bytes(request.report.devices[0])),
              static_cast<unsigned long long>(budget));
}

void test_nothing_fits(const char *path) {
  fit_request full;
  assert(full.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  const llama_dart_fit_device device = full.report.devices[0];
  ggml_backend_dev_t gpu = ggml_backend_dev_by_name(device.name);
  assert(gpu != nullptr);
  assert(device.total_bytes > 0);

  ggml_backend_dev_t devices[] = {gpu, nullptr};
  fit_request request;
  request.model.devices = devices;
  request.margins[0] = static_cast<size_t>(device.total_bytes);
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.model.n_gpu_layers == 0);
  assert(request.report.devices[0].model_bytes == 0);
}

void test_context_fit(const char *path) {
  fit_request full;
  full.context.n_ctx = 0;
  assert(full.run(path, 512) == LLAMA_DART_FIT_STATUS_SUCCESS);
  const llama_dart_fit_device device = full.report.devices[0];
  const uint64_t weights = device.model_bytes;

  fit_request request;
  request.context.n_ctx = 0;
  set_budget(request, device, weights + 160 * kMiB);
  assert(request.run(path, 512) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.context.n_ctx >= 512);
  assert(request.context.n_ctx < kTrainedContext);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(projected_bytes(request.report.devices[0]) <= weights + 160 * kMiB);

  fit_request kept;
  kept.context.n_ctx = 0;
  set_budget(kept, device, weights + 160 * kMiB);
  assert(kept.run(path, UINT32_MAX) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(kept.context.n_ctx == 0 || kept.context.n_ctx == kTrainedContext);
}

// A preset layer count is refused only when the load does not fit with it.
void test_preset_layers(const char *path) {
  fit_request full;
  assert(full.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  const llama_dart_fit_device device = full.report.devices[0];

  fit_request roomy;
  roomy.model.n_gpu_layers = 4;
  assert(roomy.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(roomy.model.n_gpu_layers == 4);

  fit_request request;
  request.model.n_gpu_layers = 4;
  set_budget(request, device, 1 * kMiB);
  const uint32_t n_ctx = request.context.n_ctx;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_FAILURE);
  assert(request.model.n_gpu_layers == 4);
  assert(request.context.n_ctx == n_ctx);
}

// A failed fit leaves n_ctx at 0 although it had already reduced the context.
void test_failure_keeps_context(const char *path, bool gpu) {
  ggml_backend_dev_t no_gpu[] = {nullptr};
  fit_request cpu;
  cpu.model.devices = no_gpu;
  cpu.context.n_ctx = 0;
  cpu.margins.assign(cpu.margins.size(), static_cast<size_t>(1) << 40);
  assert(cpu.run(path, 512) == LLAMA_DART_FIT_STATUS_FAILURE);
  assert(cpu.context.n_ctx == 0);
  assert(cpu.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);

  if (!gpu) {
    return;
  }
  fit_request full;
  assert(full.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  fit_request request;
  request.context.n_ctx = 0;
  request.model.n_gpu_layers = 4;
  set_budget(request, full.report.devices[0], 1 * kMiB);
  assert(request.run(path, 512) == LLAMA_DART_FIT_STATUS_FAILURE);
  assert(request.context.n_ctx == 0);
  assert(request.model.n_gpu_layers == 4);
}

// On v1.0.0 the loader counts a NextN layer that llama_model_n_layer leaves
// out, and a fit sets n_gpu_layers on the loader's scale. The report has to
// let a binding tell a fit that kept the last layer on the CPU from a full
// offload.
void test_nextn_layers(const char *path) {
  fit_request full;
  assert(full.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(full.report.n_layers == kBlocks);
  assert(full.report.n_layers_skipped == 1);
  const llama_dart_fit_device device = full.report.devices[0];
  const int32_t all = static_cast<int32_t>(full.report.n_layers +
                                           full.report.n_layers_skipped) + 1;
  bool kept_one_layer_back = false;
  for (int percent = 99; percent >= 60; --percent) {
    fit_request request;
    set_budget(request, device, projected_bytes(device) * percent / 100);
    assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
    const int32_t layers = request.model.n_gpu_layers;
    assert(layers <= all);
    if (layers > 0 && layers < all &&
        layers >= static_cast<int32_t>(request.report.n_layers) + 1) {
      kept_one_layer_back = true;
    }
  }
  assert(kept_one_layer_back);
}

int g_sentinel;

void sentinel_logger(ggml_log_level, const char *, void *user_data) {
  assert(user_data == &g_sentinel);
}

// Fits that overlap each other must put back the logger that they found. The
// installed logger is shared state: without the fit lock the fits take each
// other's logger for the original, or restore it twice.
void test_concurrent(const char *path) {
  llama_log_set(sentinel_logger, &g_sentinel);
  std::vector<std::thread> fits;
  for (int i = 0; i < 4; ++i) {
    fits.emplace_back([path] {
      for (int run = 0; run < 3; ++run) {
        fit_request request;
        assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
        assert(request.report.n_layers == kBlocks);
      }
    });
  }
  for (std::thread &fit : fits) {
    fit.join();
  }
  ggml_log_callback callback = nullptr;
  void *user_data = nullptr;
  llama_log_get(&callback, &user_data);
  assert(callback == sentinel_logger && user_data == &g_sentinel);
  llama_dart_set_log_level(4);
}

struct log_counts {
  std::atomic<long> lines{0};
  std::atomic<long> info{0};
};

struct fit_log_gate {
  std::atomic<bool> measuring{false};
  std::atomic<bool> loader_done{false};
};

log_counts g_loader_counts;
thread_local bool g_on_loader_thread = false;
thread_local fit_log_gate *g_fit_log_gate = nullptr;

void counting_logger(ggml_log_level level, const char *text, void *) {
  if (g_fit_log_gate != nullptr && std::strstr(text, "loaded meta data with") &&
      !g_fit_log_gate->measuring.exchange(true)) {
    while (!g_fit_log_gate->loader_done.load()) {
      std::this_thread::yield();
    }
  }
  if (g_on_loader_thread) {
    g_loader_counts.lines.fetch_add(1);
    if (level == GGML_LOG_LEVEL_INFO) {
      g_loader_counts.info.fetch_add(1);
    }
  }
}

void load_vocabulary_only(const char *path) {
  llama_model_params params = llama_model_default_params();
  params.vocab_only = true;
  llama_model *model = llama_model_load_from_file(path, params);
  assert(model != nullptr);
  llama_model_free(model);
}

// A fit lowers the log levels of its own thread only. Another thread that
// loads a model meanwhile must get its INFO lines as they are: llama.cpp's own
// scope lowered the lines of every thread. Needs the carried patch 0007.
void test_other_threads_keep_log_levels(const char *path) {
  llama_log_set(counting_logger, nullptr);
  g_on_loader_thread = true;
  for (int run = 0; run < 20; ++run) {
    load_vocabulary_only(path);
  }
  const long control_lines = g_loader_counts.lines.load();
  const long control_info = g_loader_counts.info.load();
  assert(control_info > 0);

  g_loader_counts.lines = 0;
  g_loader_counts.info = 0;
  fit_log_gate gate;
  std::atomic<bool> fitted{false};
  std::thread fit([&] {
    g_fit_log_gate = &gate;
    fit_request request;
    assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
    fitted.store(true);
  });
  while (!gate.measuring.load() && !fitted.load()) {
    std::this_thread::yield();
  }
  assert(gate.measuring.load());
  // The fit is stopped inside its logger, so this load overlaps the scope.
  load_vocabulary_only(path);
  gate.loader_done.store(true);
  fit.join();
  g_on_loader_thread = false;

  const long lines = g_loader_counts.lines.load();
  const long info = g_loader_counts.info.load();
  assert(lines > 0);
  // The share of INFO lines is the same as without a fit, not close to zero.
  assert(info * control_lines * 2 >= control_info * lines);
  llama_dart_set_log_level(4);
}

struct fit_probe_logs {
  int info = 0;
  int warnings = 0;
  int errors = 0;
  int info_cont = 0;
  int warn_cont = 0;
  bool inject_cont = false;
  bool injected_cont = false;
};

void fit_probe_logger(ggml_log_level level, const char *text, void *user_data) {
  auto &logs = *static_cast<fit_probe_logs *>(user_data);
  if (level == GGML_LOG_LEVEL_INFO) {
    ++logs.info;
  } else if (level == GGML_LOG_LEVEL_WARN) {
    ++logs.warnings;
  } else if (level == GGML_LOG_LEVEL_ERROR) {
    ++logs.errors;
    if (logs.inject_cont && !logs.injected_cont) {
      logs.injected_cont = true;
      ggml_log_callback forwarder = nullptr;
      void *forwarder_ud = nullptr;
      llama_log_get(&forwarder, &forwarder_ud);
      assert(forwarder != nullptr);
      forwarder(GGML_LOG_LEVEL_INFO, "probe info\n", forwarder_ud);
      forwarder(GGML_LOG_LEVEL_CONT, "probe info cont\n", forwarder_ud);
      forwarder(GGML_LOG_LEVEL_WARN, "probe warn\n", forwarder_ud);
      forwarder(GGML_LOG_LEVEL_CONT, "probe warn cont\n", forwarder_ud);
    }
  } else if (level == GGML_LOG_LEVEL_CONT && text != nullptr) {
    if (std::strstr(text, "probe info cont") != nullptr) {
      ++logs.info_cont;
    } else if (std::strstr(text, "probe warn cont") != nullptr) {
      ++logs.warn_cont;
    }
  }
}

void test_fit_probe_log_scope(const char *path) {
  fit_probe_logs logs;
  llama_log_set(fit_probe_logger, &logs);
  load_vocabulary_only(path);
  assert(logs.info > 0);

  logs = {};
  const auto params = llama_model_default_params();
  llama_model *model = llama_dart_test_load_fit_probe(path, &params);
  assert(model != nullptr);
  llama_model_free(model);
  assert(logs.info == 0);

  llama_model_kv_override overrides[2] = {};
  std::snprintf(overrides[0].key, sizeof(overrides[0].key), "llama.block_count");
  overrides[0].tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
  overrides[0].val_f64 = 1.0;
  auto warn_params = llama_model_default_params();
  warn_params.kv_overrides = overrides;
  logs = {};
  llama_model *warn_model = llama_dart_test_load_fit_probe(path, &warn_params);
  assert(warn_model != nullptr);
  llama_model_free(warn_model);
  assert(logs.info == 0);
  assert(logs.warnings > 0);

  logs = {};
  logs.inject_cont = true;
  assert(llama_dart_test_load_fit_probe("fit_api_test_missing_model.gguf",
                                        &params) == nullptr);
  assert(logs.errors > 0);
  assert(logs.info == 0);
  assert(logs.info_cont == 0);
  assert(logs.warnings > 0);
  assert(logs.warn_cont > 0);

  ggml_log_callback callback = nullptr;
  void *user_data = nullptr;
  llama_log_get(&callback, &user_data);
  assert(callback == fit_probe_logger && user_data == &logs);
  load_vocabulary_only(path);
  assert(logs.info > 0);
  llama_dart_set_log_level(4);
}

struct probe_log_gate {
  std::atomic<bool> probing{false};
  std::atomic<bool> loader_done{false};
};

thread_local probe_log_gate *g_probe_log_gate = nullptr;

void probe_xthread_logger(ggml_log_level level, const char *, void *) {
  if (g_probe_log_gate != nullptr && level == GGML_LOG_LEVEL_ERROR &&
      !g_probe_log_gate->probing.exchange(true)) {
    while (!g_probe_log_gate->loader_done.load()) {
      std::this_thread::yield();
    }
  }
  if (g_on_loader_thread) {
    g_loader_counts.lines.fetch_add(1);
    if (level == GGML_LOG_LEVEL_INFO) {
      g_loader_counts.info.fetch_add(1);
    }
  }
}

void test_fit_probe_other_thread_keeps_log_levels(const char *path) {
  llama_log_set(probe_xthread_logger, nullptr);
  g_loader_counts.lines = 0;
  g_loader_counts.info = 0;
  g_on_loader_thread = true;
  load_vocabulary_only(path);
  const long control_info = g_loader_counts.info.load();
  assert(control_info > 0);

  g_loader_counts.lines = 0;
  g_loader_counts.info = 0;
  probe_log_gate gate;
  std::atomic<bool> probe_done{false};
  std::thread probe([&] {
    g_probe_log_gate = &gate;
    const auto params = llama_model_default_params();
    assert(llama_dart_test_load_fit_probe("fit_api_test_missing_model.gguf",
                                          &params) == nullptr);
    g_probe_log_gate = nullptr;
    probe_done.store(true);
  });
  while (!gate.probing.load() && !probe_done.load()) {
    std::this_thread::yield();
  }
  assert(gate.probing.load());
  // The probe thread is blocked inside its scope, so this load overlaps it.
  load_vocabulary_only(path);
  gate.loader_done.store(true);
  probe.join();
  g_on_loader_thread = false;

  const long during_info = g_loader_counts.info.load();
  assert(during_info == control_info);
  llama_dart_set_log_level(4);
}

constexpr size_t kDriverFree = 21663ull * kMiB;
constexpr size_t kDriverTotal = 23033ull * kMiB;

struct fake_fit_gpu_info {
  const char *name;
  const char *description;
};

// Reuse the CPU backend for no-allocation graph planning on GPU-free CI hosts.
ggml_backend_device fake_fit_gpu(fake_fit_gpu_info &info) {
  const auto cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  assert(cpu != nullptr);
  ggml_backend_device device = *cpu;
  device.context = &info;
  device.iface.get_name = [](ggml_backend_dev_t dev) {
    return static_cast<fake_fit_gpu_info *>(dev->context)->name;
  };
  device.iface.get_description = [](ggml_backend_dev_t dev) {
    return static_cast<fake_fit_gpu_info *>(dev->context)->description;
  };
  device.iface.get_memory = [](ggml_backend_dev_t, size_t *free, size_t *total) {
    *free = kDriverFree;
    *total = kDriverTotal;
  };
  device.iface.get_type = [](ggml_backend_dev_t) -> enum ggml_backend_dev_type {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
  };
  device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props *props) {
    props->name = dev->iface.get_name(dev);
    props->description = dev->iface.get_description(dev);
    props->type = dev->iface.get_type(dev);
    dev->iface.get_memory(dev, &props->memory_free, &props->memory_total);
  };
  return device;
}

ggml_backend_dev_t fake_fit_cuda_device() {
  static fake_fit_gpu_info info{"CUDA0", "NVIDIA L4"};
  static ggml_backend_device device = fake_fit_gpu(info);
  return &device;
}

void test_c7_fit_report(const char *path, int64_t expected_free) {
  const llama_dart_test_fit_row rows[] = {
      {fake_fit_cuda_device(), false, kDriverTotal, kDriverFree, 4131828531ull,
       16128ull * kMiB, 288ull * kMiB},
      {nullptr, true, 34355970048ll, 29416341504ll, 1919000576ull, 6720ull * kMiB,
       289406976ull},
  };
  llama_dart_test_set_fit_rows(rows, 2);

  ggml_backend_dev_t no_gpu[] = {nullptr};
  fit_request request;
  request.model.devices = no_gpu;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  llama_dart_test_set_fit_rows(nullptr, 0);
  assert(request.report.n_devices == 1);
  assert(std::string(request.report.devices[0].name) == "CUDA0");
  assert(request.report.devices[0].free_bytes == expected_free);
  assert(request.report.devices[0].context_bytes == 16128ull * kMiB);
  assert(request.report.host.context_bytes == 6720ull * kMiB);
}

#if defined(_WIN32)
constexpr uint8_t kCudaLuid[8] = {0x5D, 0x50, 0, 0, 0x12, 0x34, 0x56, 0x78};
constexpr uint8_t kVulkanLuid[8] = {0x5E, 0x50, 0, 0, 0x12, 0x34, 0x56, 0x78};
constexpr size_t kExpectedFree = 6263ull * kMiB;
bool g_fake_kmt_fail = false;
bool g_fake_cuda_fail = false;
int g_fake_cuda_reads = 0;
int g_fake_vulkan_reads = 0;

int fake_win_cuda_init(unsigned) {
  ++g_fake_cuda_reads;
  return g_fake_cuda_fail ? 1 : 0;
}
int fake_win_cuda_count(int *count) {
  *count = 1;
  return 0;
}
int fake_win_cuda_get(int *dev, int ord) {
  *dev = ord;
  return 0;
}
int fake_win_cuda_name(char *name, int len, int) {
  std::snprintf(name, static_cast<size_t>(len), "NVIDIA L4");
  return 0;
}
int fake_win_cuda_luid(char *luid, unsigned *mask, int) {
  std::memcpy(luid, kCudaLuid, sizeof(kCudaLuid));
  *mask = 1;
  return 0;
}

long __stdcall fake_win_kmt_query(void *raw) {
  if (g_fake_kmt_fail) {
    return -1073741823L; // STATUS_UNSUCCESSFUL
  }
  auto *buf = static_cast<unsigned char *>(raw);
  uint32_t type = 0;
  std::memcpy(&type, buf + 0, sizeof(type));
  const bool vulkan = std::memcmp(buf + 4, kVulkanLuid, sizeof(kVulkanLuid)) == 0;
  assert(vulkan || std::memcmp(buf + 4, kCudaLuid, sizeof(kCudaLuid)) == 0);
  if (type == 0) {
    if (vulkan) {
      ++g_fake_vulkan_reads;
    }
    const uint32_t nb_segments = 1;
    std::memcpy(buf + 24, &nb_segments, sizeof(nb_segments));
    return 0;
  }
  if (type == 3) {
    // Independently specified D3DKMT_QUERYSTATISTICS_SEGMENT_INFORMATION offsets:
    // Result starts at byte 24: CommitLimit (+0), BytesCommitted (+8),
    // BytesResident (+16), Aperture (+40).
    const uint64_t commit_limit = 22595ull * kMiB;
    const uint64_t bytes_committed = 100ull * kMiB; // distinct from resident!
    const uint64_t bytes_resident = 16332ull * kMiB;
    const uint32_t aperture = 0;
    std::memcpy(buf + 24 + 0, &commit_limit, sizeof(commit_limit));
    std::memcpy(buf + 24 + 8, &bytes_committed, sizeof(bytes_committed));
    std::memcpy(buf + 24 + 16, &bytes_resident, sizeof(bytes_resident));
    std::memcpy(buf + 24 + 40, &aperture, sizeof(aperture));
    return 0;
  }
  return -1;
}
void test_windows_adapter_paths(const char *path) {
  static fake_fit_gpu_info vulkan_info{"Vulkan0", "Fixture Vulkan adapter"};
  static ggml_backend_device vulkan = fake_fit_gpu(vulkan_info);
  const auto cuda = fake_fit_cuda_device();
  ggml_backend_device_register(cuda);
  ggml_backend_device_register(&vulkan);

  llama_dart_test_win_dev_config config;
  config.cuda_device = cuda;
  config.vulkan_device = &vulkan;
  std::memcpy(config.vulkan_luid, kVulkanLuid, sizeof(kVulkanLuid));
  config.cuda_init = fake_win_cuda_init;
  config.cuda_device_count = fake_win_cuda_count;
  config.cuda_device_get = fake_win_cuda_get;
  config.cuda_device_name = fake_win_cuda_name;
  config.cuda_device_luid = fake_win_cuda_luid;
  config.cuda_registered_count = 1;
  config.kmt_query = fake_win_kmt_query;
  llama_dart_test_set_win_dev_config(&config);

  g_fake_cuda_fail = true;
  g_fake_cuda_reads = 0;
  g_fake_vulkan_reads = 0;
  test_cpu_only(path);
  assert(g_fake_cuda_reads == 0);
  assert(g_fake_vulkan_reads == 0);

  ggml_backend_dev_t vulkan_only[] = {&vulkan, nullptr};
  fit_request vulkan_fit;
  vulkan_fit.model.devices = vulkan_only;
  fit_probe_logs logs;
  llama_log_set(fit_probe_logger, &logs);
  assert(vulkan_fit.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(logs.info == 0);
  llama_dart_set_log_level(4);
  assert(g_fake_cuda_reads == 0);
  assert(g_fake_vulkan_reads == 2);
  assert(vulkan_fit.report.n_devices == 1);
  assert(std::string(vulkan_fit.report.devices[0].name) == "Vulkan0");
  assert(vulkan_fit.report.devices[0].free_bytes == kExpectedFree);

  g_fake_cuda_fail = false;
  ggml_backend_dev_t cuda_only[] = {cuda, nullptr};
  fit_request cuda_fit;
  cuda_fit.model.devices = cuda_only;
  assert(cuda_fit.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(g_fake_cuda_reads == 2);
  assert(g_fake_vulkan_reads == 2);
  assert(cuda_fit.report.n_devices == 1);
  assert(std::string(cuda_fit.report.devices[0].name) == "CUDA0");
  assert(cuda_fit.report.devices[0].free_bytes == kExpectedFree);

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  assert(llama_dart_ggml_backend_dev_memory(cuda, &free_bytes, &total_bytes));
  assert(free_bytes == kExpectedFree);
  assert(total_bytes == kDriverTotal);
  ggml_backend_dev_props props = {};
  assert(llama_dart_ggml_backend_dev_get_props(cuda, &props));
  assert(props.memory_free == kExpectedFree);
  test_c7_fit_report(path, kExpectedFree);

  g_fake_kmt_fail = true;
  assert(llama_dart_ggml_backend_dev_memory(cuda, &free_bytes, &total_bytes));
  assert(free_bytes == kDriverFree);
  assert(llama_dart_ggml_backend_dev_get_props(cuda, &props));
  assert(props.memory_free == kDriverFree);

  const llama_dart_test_fit_row rows[] = {
      {cuda, false, kDriverTotal, kDriverFree, 0, 0, 0},
      {nullptr, true, 34355970048ll, 29416341504ll, 0, 0, 0},
  };
  llama_dart_test_set_fit_rows(rows, 2);
  ggml_backend_dev_t no_gpu[] = {nullptr};
  fit_request strict;
  strict.model.devices = no_gpu;
  strict.context.n_ctx = 0;
  strict.report.n_layers = 777;
  assert(strict.run(path) == LLAMA_DART_STATUS_EXCEPTION);
  assert(strict.model.devices == no_gpu);
  assert(strict.context.n_ctx == 0);
  assert(strict.report.n_layers == 777);
  const char *error = llama_dart_last_error();
  assert(error != nullptr && std::string(error).find("CUDA0") != std::string::npos);
  llama_dart_test_set_fit_rows(nullptr, 0);
  g_fake_kmt_fail = false;

  llama_dart_test_set_win_dev_config(nullptr);
}
#endif

} // namespace

int main(int argc, char **argv) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
  llama_backend_init();
  llama_dart_set_log_level(4);
  const char *path = argc > 1 ? argv[1] : "fit_api_test_model.gguf";
  const char *nextn_path = "fit_api_test_nextn_model.gguf";
  if (argc <= 1) {
    make_model(path);
  }

  test_contract(path);
  test_unreadable_model();
  test_cpu_only(path);
  const bool gpu = has_gpu(path);
  if (gpu) {
    test_everything_fits(path);
    test_partial_fit(path);
    test_nothing_fits(path);
    test_context_fit(path);
    test_preset_layers(path);
#if LLAMADART_FIT_HAS_NGL_SKIP
    make_model(nextn_path, true);
    test_nextn_layers(nextn_path);
    std::remove(nextn_path);
#endif
  }
  test_failure_keeps_context(path, gpu);
  test_concurrent(path);
  test_fit_probe_log_scope(path);
  test_fit_probe_other_thread_keeps_log_levels(path);
  test_other_threads_keep_log_levels(path);
#if defined(_WIN32)
  test_windows_adapter_paths(path);
#else
  test_c7_fit_report(path, kDriverFree);
#endif
  test_missing_measured_device(path);
  std::printf("fit api test passed (%s)\n", gpu ? "gpu" : "cpu only");
  if (argc <= 1) {
    std::remove(path);
  }
  return 0;
}
