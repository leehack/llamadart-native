#include "llama_dart_wrapper.h"
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

// A llama decoder of about 36 MiB with a long trained context, so that its
// weights and its cache are large enough to be fitted into a budget of a few
// tens of MiB.
void make_model(const char *path) {
  gguf_context *gguf = gguf_init_empty();
  const auto set = [&](const char *key, uint32_t value) {
    gguf_set_val_u32(gguf, (std::string("llama.") + key).c_str(), value);
  };
  gguf_set_val_str(gguf, "general.architecture", "llama");
  gguf_set_val_str(gguf, "general.name", "llamadart fit test");
  set("context_length", kTrainedContext);
  set("embedding_length", kEmbedding);
  set("block_count", kBlocks);
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
}

void test_everything_fits(const char *path) {
  fit_request request;
  assert(request.run(path) == LLAMA_DART_FIT_STATUS_SUCCESS);
  assert(request.model.n_gpu_layers == llama_model_default_params().n_gpu_layers);
  assert(request.context.n_ctx == 2048);
  assert(request.report.n_layers == kBlocks);
  const llama_dart_fit_device &device = request.report.devices[0];
  assert(device.name[0] != '\0');
  assert(device.free_bytes > 0 && device.total_bytes >= device.free_bytes);
  assert(projected_bytes(device) > 0);
  assert(device.model_bytes >= 30 * kMiB);
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

// Fits that overlap each other and a log level change must neither crash nor
// disturb each other's result.
void test_concurrent(const char *path) {
  std::atomic<bool> stop{false};
  std::thread logger([&stop] {
    while (!stop.load()) {
      llama_dart_set_log_level(3);
    }
  });
  std::vector<std::thread> fits;
  for (int i = 0; i < 3; ++i) {
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
  stop.store(true);
  logger.join();
}

} // namespace

int main(int argc, char **argv) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
  llama_backend_init();
  llama_dart_set_log_level(4);
  const char *path = argc > 1 ? argv[1] : "fit_api_test_model.gguf";
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
  } else {
    test_cpu_only(path);
  }
  test_concurrent(path);
  std::printf("fit api test passed (%s)\n", gpu ? "gpu" : "cpu only");
  if (argc <= 1) {
    std::remove(path);
  }
  return 0;
}
