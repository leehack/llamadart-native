#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if !defined(LLAMADART_TEST_CPU_BACKEND)
#include "ggml-cpu.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

float rounded_f16(float value) {
  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

class Graph {
 public:
  explicit Graph(ggml_backend_t backend) : backend_(backend) {
    ggml_init_params params = {1024 * 1024, nullptr, true};
    ctx = ggml_init(params);
    require(ctx != nullptr, "context allocation failed");
  }

  ~Graph() {
    ggml_backend_buffer_free(buffer_);
    ggml_free(ctx);
  }

  void allocate(ggml_tensor *output) {
    graph_ = ggml_new_graph(ctx);
    require(graph_ != nullptr, "graph allocation failed");
    ggml_build_forward_expand(graph_, output);
    buffer_ = ggml_backend_alloc_ctx_tensors(ctx, backend_);
    require(buffer_ != nullptr, "tensor buffer allocation failed");
  }

  void compute() {
    require(ggml_backend_graph_compute(backend_, graph_) == GGML_STATUS_SUCCESS,
            "CPU graph computation failed");
  }

  ggml_context *ctx = nullptr;

 private:
  ggml_backend_t backend_;
  ggml_cgraph *graph_ = nullptr;
  ggml_backend_buffer_t buffer_ = nullptr;
};

void check_result(float actual, double expected, double tolerance,
                  const char *name, int width) {
  const double error = std::fabs(static_cast<double>(actual) - expected);
  if (!std::isfinite(actual) || !std::isfinite(expected) ||
      !std::isfinite(error) || error > tolerance) {
    std::fprintf(stderr,
                 "FAIL: %s width=%d expected=%.17g actual=%.9g tolerance=%.9g\n",
                 name, width, expected, actual, tolerance);
    std::exit(1);
  }
}

enum class Pattern { positive, cancellation, large_product, maximum, subnormal };

void dot_case(ggml_backend_t backend, int width, float magnitude,
              Pattern pattern, ggml_type weight_type, const char *name) {
  Graph graph(backend);
  auto *k = ggml_new_tensor_2d(graph.ctx, weight_type, width, 1);
  auto *q = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, width, 1);
  auto *out = ggml_mul_mat(graph.ctx, k, q);
  ggml_mul_mat_set_prec(out, GGML_PREC_F32);
  graph.allocate(out);

  std::vector<float> keys(width), queries(width);
  std::vector<ggml_fp16_t> half_keys(width);
  double reference = 0.0;
  double absolute_sum = 0.0;
  for (int i = 0; i < width; ++i) {
    float kv = magnitude * (1.0f + 0.1f * (i % 5));
    float qv = magnitude * (1.0f + 0.1f * ((i * 3) % 5));
    if (pattern == Pattern::cancellation) {
      kv = magnitude;
      qv = i < width / 2 ? magnitude : -magnitude;
    } else if (pattern == Pattern::large_product) {
      kv = qv = 256.0f;
    } else if (pattern == Pattern::maximum) {
      kv = qv = 65504.0f;
    } else if (pattern == Pattern::subnormal) {
      kv = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(1 + i % 7));
      qv = i % 2 == 0 ? 1.0f : -1.0f;
    }
    half_keys[i] = ggml_fp32_to_fp16(kv);
    keys[i] = kv;
    queries[i] = qv;
    const double actual_k = weight_type == GGML_TYPE_F16
                                ? ggml_fp16_to_fp32(half_keys[i]) : keys[i];
    // The F16 CPU type trait rounds the F32 query before its dot product.
    const double actual_q = weight_type == GGML_TYPE_F16
                                ? rounded_f16(queries[i]) : queries[i];
    const double product = actual_k * actual_q;
    reference += product;
    absolute_sum += std::fabs(product);
  }
  if (weight_type == GGML_TYPE_F16) {
    ggml_backend_tensor_set(k, half_keys.data(), 0, half_keys.size() * sizeof(ggml_fp16_t));
  } else {
    ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(float));
  }
  ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
  graph.compute();
  float actual = 0.0f;
  ggml_backend_tensor_get(out, &actual, 0, sizeof(actual));
  check_result(actual, reference, 1e-7 + 2e-6 * absolute_sum, name, width);
}

void attention_case(ggml_backend_t backend, bool flash, float magnitude) {
  constexpr int width = 64;
  constexpr int values = 64;
  Graph graph(backend);
  auto *q = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, width, 1);
  auto *k = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, width, 2);
  auto *v = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16,
                               flash ? values : 2, flash ? 2 : values);
  const float scale = 1.0f / std::sqrt(static_cast<float>(width));
  ggml_tensor *out;
  if (flash) {
    out = ggml_flash_attn_ext(graph.ctx, q, k, v, nullptr, scale, 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
  } else {
    auto *scores = ggml_mul_mat(graph.ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    auto *probabilities = ggml_soft_max_ext(graph.ctx, scores, nullptr, scale, 0.0f);
    out = ggml_mul_mat(graph.ctx, v, probabilities);
  }
  graph.allocate(out);

  std::vector<float> queries(width);
  std::vector<ggml_fp16_t> keys(2 * width), value_data(2 * values);
  double score = 0.0;
  for (int i = 0; i < width; ++i) {
    queries[i] = magnitude * (1.0f + 0.1f * ((i * 3) % 5));
    keys[i] = ggml_fp32_to_fp16(magnitude * (1.0f + 0.1f * (i % 5)));
    keys[width + i] = ggml_fp32_to_fp16(0.0f);
    score += static_cast<double>(ggml_fp16_to_fp32(keys[i])) * rounded_f16(queries[i]);
  }
  const double probability0 = 1.0 / (1.0 + std::exp(-score * scale));
  const double probability1 = 1.0 - probability0;
  std::vector<double> expected(values);
  for (int i = 0; i < values; ++i) {
    const float v0 = 1.0f + (i % 7) * 0.125f;
    const float v1 = 2.0f + (i % 5) * 0.25f;
    value_data[flash ? i : 2 * i] = ggml_fp32_to_fp16(v0);
    value_data[flash ? values + i : 2 * i + 1] = ggml_fp32_to_fp16(v1);
    // Ordinary F16 V matmul also rounds its probability input to F16.
    const double p0 = flash ? probability0 : rounded_f16(static_cast<float>(probability0));
    const double p1 = flash ? probability1 : rounded_f16(static_cast<float>(probability1));
    expected[i] = v0 * p0 + v1 * p1;
  }
  ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
  ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(v, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
  graph.compute();
  std::vector<float> actual(values);
  ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
  for (int i = 0; i < values; ++i) {
    check_result(actual[i], expected[i], 1e-3, flash ? "flash-attention" : "attention", width);
  }
}

}  // namespace

int main() {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  ggml_backend_reg_t reg = ggml_backend_load(LLAMADART_TEST_CPU_BACKEND);
  std::printf("CPU module: %s\n", LLAMADART_TEST_CPU_BACKEND);
#else
  ggml_backend_reg_t reg = ggml_backend_cpu_reg();
  std::printf("CPU module: statically linked\n");
#endif
  require(reg != nullptr, "requested CPU module did not load");
  require(ggml_backend_reg_dev_count(reg) == 1, "unexpected CPU module device count");
  auto device = ggml_backend_reg_dev_get(reg, 0);
  require(device != nullptr && ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_CPU,
          "requested module did not provide a CPU device");
  auto backend = ggml_backend_dev_init(device, nullptr);
  require(backend != nullptr, "CPU backend initialization failed");
  auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
      ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
  require(set_threads != nullptr, "CPU thread setter unavailable");
  set_threads(backend, 1);
  auto get_features = reinterpret_cast<ggml_backend_get_features_t>(
      ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features"));
  require(get_features != nullptr, "CPU features unavailable");
  std::printf("CPU features:");
  for (auto *feature = get_features(reg); feature->name != nullptr; ++feature) {
    std::printf(" %s=%s", feature->name, feature->value);
  }
  std::printf("\n");

  for (int magnitude = 30; magnitude <= 100; magnitude += 10) {
    dot_case(backend, 64, static_cast<float>(magnitude), Pattern::positive,
             GGML_TYPE_F16, "upstream-positive");
  }
  const int widths[] = {1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129};
  for (int width : widths) {
    dot_case(backend, width, 100.0f, Pattern::positive, GGML_TYPE_F16, "tail-positive");
    dot_case(backend, width, 100.0f, Pattern::cancellation, GGML_TYPE_F16, "cancellation");
    dot_case(backend, width, 0.0f, Pattern::subnormal, GGML_TYPE_F16, "subnormal");
  }
  dot_case(backend, 64, 1.0f, Pattern::positive, GGML_TYPE_F16, "low-positive");
  dot_case(backend, 64, 0.0f, Pattern::positive, GGML_TYPE_F16, "zero");
  dot_case(backend, 32, 0.0f, Pattern::large_product, GGML_TYPE_F16, "large-product");
  dot_case(backend, 64, 0.0f, Pattern::maximum, GGML_TYPE_F16, "maximum-finite");
  dot_case(backend, 64, 100.0f, Pattern::positive, GGML_TYPE_F32, "f32-control");
  for (float magnitude : {0.0f, 1.0f, 80.0f, 100.0f}) {
    attention_case(backend, false, magnitude);
    attention_case(backend, true, magnitude);
  }
  ggml_backend_free(backend);
#if defined(LLAMADART_TEST_CPU_BACKEND)
  ggml_backend_unload(reg);
#endif
  std::printf("PASS: finite F16 dot, tails, controls and ordinary/flash attention\n");
  return 0;
}
