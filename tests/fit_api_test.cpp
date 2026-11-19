#include "llama_dart_wrapper.h"
#include "llama_dart_fit_test_hooks.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
  fit_request request;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.report.n_devices == 0);
  assert(request.report.n_layers == kBlocks);
  assert(request.report.n_ctx_train == kTrainedContext);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 2048);
  assert(std::string(request.report.host.name) == "Host");
  assert(request.report.host.total_bytes > 0 && request.report.host.free_bytes > 0);
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

  fit_request request;
  set_budget(request, device, 1 * kMiB);
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

log_counts g_loader_counts;
thread_local bool g_on_loader_thread = false;

void counting_logger(ggml_log_level level, const char *, void *) {
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
  std::atomic<bool> fitting{false};
  std::atomic<bool> fitted{false};
  std::thread fit([&] {
    fitting.store(true);
    for (int run = 0; run < 3; ++run) {
      fit_request request;
      assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
    }
    fitted.store(true);
  });
  while (!fitting.load()) {
    std::this_thread::yield();
  }
  while (!fitted.load()) {
    load_vocabulary_only(path);
  }
  fit.join();
  g_on_loader_thread = false;

  const long lines = g_loader_counts.lines.load();
  const long info = g_loader_counts.info.load();
  assert(lines > 0);
  // The share of INFO lines is the same as without a fit, not close to zero.
  assert(info * control_lines * 2 >= control_info * lines);
  llama_dart_set_log_level(4);
}

void test_commit_headroom_contract(const char *path) {
  struct guard_reset {
    ~guard_reset() { llama_dart_test_set_fit_commit_config(nullptr); }
  } reset;

  const std::vector<llama_dart_test_fit_row> c7_rows = {
      {"CUDA0", true, false, 24152440832ll, 22715891712ll, 4131828531ull,
       16911433728ull, 301989888ull},
      {"Host", false, true, 34355970048ll, 29416341504ll, 1919000576ull,
       7046430720ull, 289406976ull},
  };
  const uint64_t c7_total_bytes =
      4131828531ull + 16911433728ull + 301989888ull + 1919000576ull +
      7046430720ull + 289406976ull;

  const auto run_with_config = [&](const llama_dart_test_fit_commit_config &cfg,
                                   bool with_report, int32_t preset_ngl = -1) {
    llama_dart_test_set_fit_commit_config(&cfg);
    fit_request req;
    req.model.n_gpu_layers = preset_ngl;
    req.context.n_ctx = 2048;
    req.report.n_layers = 777;
    const int32_t rc = llama_dart_fit_params(
        path, &req.model, &req.context, req.split.data(), req.overrides.data(),
        req.margins.data(), UINT32_MAX, with_report ? &req.report : nullptr);
    return std::make_pair(rc, req);
  };

  // 1. Surplus passes.
  {
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = c7_rows.data();
    cfg.n_rows = c7_rows.size();
    cfg.commit_limit_pages = (c7_total_bytes / 4096ull) + 1024ull;
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_SUCCESS);
    assert(req.report.n_devices == 1);
  }

  // 2. Exact equality passes.
  {
    const std::vector<llama_dart_test_fit_row> exact_rows = {
        {"CUDA0", true, false, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
        {"Host", false, true, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = exact_rows.data();
    cfg.n_rows = exact_rows.size();
    cfg.commit_limit_pages = 10;
    cfg.commit_total_pages = 8;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_SUCCESS);
  }

  // 3. 1-byte deficit fails with FAILURE, both with report and report==nullptr,
  // leaving mparams/cparams/report untouched and recording commit error.
  for (bool with_report : {true, false}) {
    const std::vector<llama_dart_test_fit_row> deficit_rows = {
        {"CUDA0", true, false, 1000000ll, 1000000ll, 4097ull, 0ull, 0ull},
        {"Host", false, true, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = deficit_rows.data();
    cfg.n_rows = deficit_rows.size();
    cfg.commit_limit_pages = 10;
    cfg.commit_total_pages = 8;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, with_report, 4);
    assert(rc == LLAMA_DART_FIT_STATUS_FAILURE);
    assert(req.model.n_gpu_layers == 4);
    assert(req.context.n_ctx == 2048);
    assert(req.report.n_layers == 777);
    const char *err = llama_dart_last_error();
    assert(err != nullptr && std::string(err).find("commit") != std::string::npos);
  }

  // 4. Zero headroom fails with FAILURE.
  {
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = c7_rows.data();
    cfg.n_rows = c7_rows.size();
    cfg.commit_limit_pages = 100;
    cfg.commit_total_pages = 100;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_FAILURE);
  }

  // 5. C4 recorded numbers pass under C7's commit headroom.
  {
    const std::vector<llama_dart_test_fit_row> c4_rows = {
        {"CUDA0", true, false, 24152440832ll, 1630961664ll, 0ull, 0ull,
         33554432ull},
        {"Host", false, true, 34355970048ll, 29240696832ll, 6050807808ull,
         4563402752ull, 356515840ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = c4_rows.data();
    cfg.n_rows = c4_rows.size();
    cfg.commit_limit_pages = (33788ull * kMiB) / 4096ull;
    cfg.commit_total_pages = (6212ull * kMiB) / 4096ull;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_SUCCESS);
  }

  // 6. Unreadable counter, zero page size, counter overflow, projection overflow
  // all return LLAMA_DART_STATUS_EXCEPTION (not FAILURE).
  {
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = c7_rows.data();
    cfg.n_rows = c7_rows.size();
    cfg.counter_ok = false;
    cfg.counter_error = "injected K32GetPerformanceInfo failure";
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_STATUS_EXCEPTION);
    assert(req.report.n_layers == 777);
  }
  {
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = c7_rows.data();
    cfg.n_rows = c7_rows.size();
    cfg.commit_limit_pages = UINT64_MAX;
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_STATUS_EXCEPTION);
  }
  {
    const std::vector<llama_dart_test_fit_row> overflow_rows = {
        {"CUDA0", true, false, 1000000ll, 1000000ll, UINT64_MAX, 1ull, 0ull},
        {"Host", false, true, 1000000ll, 1000000ll, 0ull, 0ull, 0ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = overflow_rows.data();
    cfg.n_rows = overflow_rows.size();
    cfg.commit_limit_pages = 1000;
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_STATUS_EXCEPTION);
  }

  // 7. CUDA beyond output slot 16 (17 devices) is still summed.
  {
    std::vector<std::string> names;
    std::vector<llama_dart_test_fit_row> many_rows;
    names.reserve(17);
    for (int i = 0; i < 17; ++i) {
      names.push_back("CUDA" + std::to_string(i));
    }
    for (int i = 0; i < 17; ++i) {
      many_rows.push_back(
          {names[i].c_str(), true, false, 1000000ll, 1000000ll,
           i == 16 ? 8192ull : 0ull, 0ull, 0ull});
    }
    many_rows.push_back(
        {"Host", false, true, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull});
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = many_rows.data();
    cfg.n_rows = many_rows.size();
    cfg.commit_limit_pages = 2; // 8192 bytes < 12288 bytes
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_FAILURE);
  }

  // 8. Multiple CUDA rows (CUDA0 + CUDA1 + Host) are summed.
  {
    const std::vector<llama_dart_test_fit_row> multi_rows = {
        {"CUDA0", true, false, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
        {"CUDA1", true, false, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
        {"Host", false, true, 1000000ll, 1000000ll, 4096ull, 0ull, 0ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = true;
    cfg.override_measurement = true;
    cfg.rows = multi_rows.data();
    cfg.n_rows = multi_rows.size();
    cfg.commit_limit_pages = 2; // 8192 bytes < 12288 bytes
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_FAILURE);
  }

  // 9. Vulkan-only / CPU-only control does not enforce WDDM CUDA commit guard.
  {
    const std::vector<llama_dart_test_fit_row> vulkan_rows = {
        {"Vulkan0", true, false, 1000000ll, 1000000ll, 1000000ull, 0ull, 0ull},
        {"Host", false, true, 1000000ll, 1000000ll, 1000000ull, 0ull, 0ull},
    };
    llama_dart_test_fit_commit_config cfg;
    cfg.enabled = true;
    cfg.force_wddm_cuda_selected = false;
    cfg.override_measurement = true;
    cfg.rows = vulkan_rows.data();
    cfg.n_rows = vulkan_rows.size();
    cfg.commit_limit_pages = 1;
    cfg.commit_total_pages = 0;
    cfg.page_size = 4096;
    auto [rc, req] = run_with_config(cfg, true);
    assert(rc == LLAMA_DART_FIT_STATUS_SUCCESS);
  }
}

#if defined(_WIN32)
namespace {
int fake_win_cuda_init(unsigned) { return 0; }
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
  static const unsigned char kLuid[8] = {0x5D, 0x50, 0, 0, 0x12, 0x34, 0x56, 0x78};
  std::memcpy(luid, kLuid, 8);
  *mask = 1;
  return 0;
}

bool g_fake_kmt_fail = false;

long __stdcall fake_win_kmt_query(void *raw) {
  if (g_fake_kmt_fail) {
    return -1073741823L; // STATUS_UNSUCCESSFUL
  }
  auto *buf = static_cast<unsigned char *>(raw);
  uint32_t type = 0;
  std::memcpy(&type, buf + 0, sizeof(type));
  if (type == 0) {
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
} // namespace

void test_windows_fake_device_kmt_and_commit(const char *path) {
  // Call the real Windows system commit reader once.
  const uint64_t real_commit =
      llama_dart_test_read_system_commit_available_bytes();
  assert(real_commit > 0);

  struct guard_reset {
    ~guard_reset() {
      g_fake_kmt_fail = false;
      llama_dart_test_set_win_dev_config(nullptr);
      llama_dart_test_set_fit_commit_config(nullptr);
    }
  } reset;

  const size_t kDriverFree = 21663ull * kMiB;
  const size_t kDriverTotal = 23033ull * kMiB;
  const size_t kExpectedFree = 6263ull * kMiB; // 22595 - 16332

  llama_dart_test_win_dev_config win_cfg;
  win_cfg.enabled = true;
  win_cfg.name = "CUDA0";
  win_cfg.description = "NVIDIA L4";
  win_cfg.type = GGML_BACKEND_DEVICE_TYPE_GPU;
  win_cfg.memory_free = kDriverFree;
  win_cfg.memory_total = kDriverTotal;
  win_cfg.has_cuda_api = true;
  win_cfg.cuda_init = fake_win_cuda_init;
  win_cfg.cuda_device_count = fake_win_cuda_count;
  win_cfg.cuda_device_get = fake_win_cuda_get;
  win_cfg.cuda_device_name = fake_win_cuda_name;
  win_cfg.cuda_device_luid = fake_win_cuda_luid;
  win_cfg.cuda_registered_count = 1;
  win_cfg.kmt_query = fake_win_kmt_query;
  llama_dart_test_set_win_dev_config(&win_cfg);

  // 1. Public dev_memory and dev_get_props return corrected minimum.
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  assert(llama_dart_ggml_backend_dev_memory(nullptr, &free_bytes, &total_bytes));
  assert(free_bytes == kExpectedFree);
  assert(total_bytes == kDriverTotal);

  ggml_backend_dev_props props = {};
  assert(llama_dart_ggml_backend_dev_get_props(nullptr, &props));
  assert(props.memory_free == kExpectedFree);

  // 2. Strict fitted report path returns corrected minimum.
  const std::vector<llama_dart_test_fit_row> rows = {
      {"CUDA0", true, false, static_cast<int64_t>(kDriverTotal),
       static_cast<int64_t>(kDriverFree), 0ull, 0ull, 32ull * kMiB},
      {"Host", false, true, 32768ll * static_cast<int64_t>(kMiB),
       28000ll * static_cast<int64_t>(kMiB), 1024ull * kMiB, 0ull, 0ull},
  };
  llama_dart_test_fit_commit_config fit_cfg;
  fit_cfg.enabled = true;
  fit_cfg.force_wddm_cuda_selected = true;
  fit_cfg.override_measurement = true;
  fit_cfg.rows = rows.data();
  fit_cfg.n_rows = rows.size();
  fit_cfg.commit_limit_pages = (32768ull * kMiB) / 4096ull;
  fit_cfg.commit_total_pages = 0;
  fit_cfg.page_size = 4096;
  llama_dart_test_set_fit_commit_config(&fit_cfg);

  fit_request req;
  assert(llama_dart_fit_params(path, &req.model, &req.context, req.split.data(),
                               req.overrides.data(), req.margins.data(),
                               UINT32_MAX, &req.report) ==
         LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(req.report.n_devices == 1);
  assert(req.report.devices[0].free_bytes == static_cast<int64_t>(kExpectedFree));

  // 3. When KMT fails: listings fall back to driver's free_bytes, strict fit
  // refuses with LLAMA_DART_STATUS_EXCEPTION.
  g_fake_kmt_fail = true;
  assert(llama_dart_ggml_backend_dev_memory(nullptr, &free_bytes, &total_bytes));
  assert(free_bytes == kDriverFree);
  assert(llama_dart_ggml_backend_dev_get_props(nullptr, &props));
  assert(props.memory_free == kDriverFree);
  assert(llama_dart_fit_params(path, &req.model, &req.context, req.split.data(),
                               req.overrides.data(), req.margins.data(),
                               UINT32_MAX, &req.report) ==
         LLAMA_DART_STATUS_EXCEPTION);
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
  } else {
    test_cpu_only(path);
  }
  test_failure_keeps_context(path, gpu);
  test_concurrent(path);
  test_other_threads_keep_log_levels(path);
  test_commit_headroom_contract(path);
#if defined(_WIN32)
  test_windows_fake_device_kmt_and_commit(path);
#endif
  std::printf("fit api test passed (%s)\n", gpu ? "gpu" : "cpu only");
  if (argc <= 1) {
    std::remove(path);
  }
  return 0;
}
