#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#if !defined(LLAMADART_TEST_CPU_BACKEND)
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int failures = 0;

void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

float rounded_f16(float value) {
  return ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
}

// Rounds to the nearest F16, ties to even, overflowing to infinity. Products
// and sums of the F16 test inputs below are exact in double, so this models one
// F16 multiply-add or add.
double round_f16(double value) {
  if (!std::isfinite(value) || value == 0.0) {
    return value;
  }
  int exponent;
  std::frexp(value, &exponent);
  const int quantum = std::max(exponent - 11, -24);
  const double rounded = std::ldexp(std::nearbyint(std::ldexp(value, -quantum)), quantum);
  return std::fabs(rounded) > 65504.0 ? std::copysign(INFINITY, value) : rounded;
}

// Whether the CPU module multiplies and adds F16 dot products in F16: FP16
// vector arithmetic without SVE (upstream widens to F32 under SVE).
bool f16_sums = false;

// Patch 0005 keeps upstream's single-column F16 dot where it is finite: four
// accumulators of eight lanes take one multiply-add per 32-element step, for
// at most 64 steps; then accumulators 0+2 and 1+3 and the two results add in
// F16, and each lane adds into F32. The lanes reduce as (0+4 + 1+5) + (2+6 +
// 3+7) in F32 and elements past the last full step add in double. A nonfinite
// result (returned as such) is recomputed in F32 by the kernel.
constexpr int dot_step = 32;
constexpr int dot_block_steps = 64;
constexpr int dot_block = dot_step * dot_block_steps;

double blocked_f16_dot(const std::vector<double> &k, const std::vector<double> &q) {
  const int n = static_cast<int>(k.size());
  const int np = n - n % dot_step;
  float lanes[8] = {};
  for (int start = 0; start < np; start += dot_block) {
    const int end = std::min(np, start + dot_block);
    double sums[dot_step] = {};
    for (int i = start; i < end; ++i) {
      sums[i % dot_step] = round_f16(sums[i % dot_step] + k[i] * q[i]);
    }
    for (int lane = 0; lane < 8; ++lane) {
      const double sum02 = round_f16(sums[lane] + sums[16 + lane]);
      const double sum13 = round_f16(sums[8 + lane] + sums[24 + lane]);
      lanes[lane] += static_cast<float>(round_f16(sum02 + sum13));
    }
  }
  float pairs[4];
  for (int lane = 0; lane < 4; ++lane) {
    pairs[lane] = lanes[lane] + lanes[4 + lane];
  }
  double sum = (pairs[0] + pairs[1]) + (pairs[2] + pairs[3]);
  for (int i = np; i < n; ++i) {
    sum += k[i] * q[i];
  }
  return sum;
}

// The F32 recompute of a nonfinite row, in the order of the first version of
// 0005: fused F32 multiply-adds in four accumulators of eight lanes per
// 32-element step, then single steps of eight into accumulator 0, the
// accumulators added in order, the low and high four lanes added, the four
// results added pairwise, and a double tail.
float f32_recompute_dot(const std::vector<double> &k, const std::vector<double> &q) {
  const int n = static_cast<int>(k.size());
  const int np = n - n % dot_step;
  const int np8 = n - n % 8;
  float acc[4][8] = {};
  int i = 0;
  for (; i < np8; ++i) {
    float &lane = i < np ? acc[(i % dot_step) / 8][i % 8] : acc[0][i % 8];
    lane = std::fmaf(static_cast<float>(k[i]), static_cast<float>(q[i]), lane);
  }
  for (int j = 1; j < 4; ++j) {
    for (int lane = 0; lane < 8; ++lane) {
      acc[0][lane] += acc[j][lane];
    }
  }
  float halves[4];
  for (int lane = 0; lane < 4; ++lane) {
    halves[lane] = acc[0][lane] + acc[0][4 + lane];
  }
  double sum = (halves[0] + halves[1]) + (halves[2] + halves[3]);
  for (; i < n; ++i) {
    sum += k[i] * q[i];
  }
  return static_cast<float>(sum);
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

bool check_result(float actual, double expected, double tolerance,
                  const char *name, int width, int cols) {
  const double error = std::fabs(static_cast<double>(actual) - expected);
  bool pass = std::isfinite(actual) && std::isfinite(expected) &&
              std::isfinite(error) && error <= tolerance;
  if (std::isnan(expected)) {
    pass = std::isnan(actual);
  } else if (std::isinf(expected)) {
    pass = actual == expected;
  }
  if (!pass) {
    std::fprintf(stderr,
                 "FAIL: %s width=%d cols=%d expected=%.17g actual=%.9g tolerance=%.9g\n",
                 name, width, cols, expected, actual, tolerance);
    ++failures;
    return false;
  }
  return true;
}

// Sixteen identical weight rows times `cols` identical query columns. One
// column takes the vector dot kernel; eight or more take llamafile GEMM where
// it is built. NaN padding after each weight row fails any read past the row,
// unless a nonfinite result is recomputed: without padding such a read takes
// the next row's finite values instead. One F16 column must match
// blocked_f16_dot exactly where that is finite and f32_recompute_dot exactly
// where it is not; everything else must match the exact sum.
void values_case(ggml_backend_t backend, const std::vector<float> &key_values,
                 const std::vector<float> &query_values, ggml_type weight_type,
                 const char *name, int cols = 1, int padding = 8) {
  constexpr int rows = 16;
  const int width = static_cast<int>(key_values.size());
  const int stride = width + padding;
  Graph graph(backend);
  auto *padded = ggml_new_tensor_2d(graph.ctx, weight_type, stride, rows);
  auto *k = ggml_view_2d(graph.ctx, padded, width, rows, padded->nb[1], 0);
  auto *q = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, width, cols);
  auto *out = ggml_mul_mat(graph.ctx, k, q);
  ggml_mul_mat_set_prec(out, GGML_PREC_F32);
  graph.allocate(out);

  std::vector<float> keys(static_cast<size_t>(stride) * rows, NAN);
  std::vector<ggml_fp16_t> half_keys(keys.size(), ggml_fp32_to_fp16(NAN));
  std::vector<float> queries(static_cast<size_t>(width) * cols);
  std::vector<double> actual_k(width), actual_q(width);
  double reference = 0.0;
  double absolute_sum = 0.0;
  for (int i = 0; i < width; ++i) {
    const float kv = key_values[i];
    const float qv = query_values[i];
    for (int r = 0; r < rows; ++r) {
      keys[static_cast<size_t>(r) * stride + i] = kv;
      half_keys[static_cast<size_t>(r) * stride + i] = ggml_fp32_to_fp16(kv);
    }
    for (int c = 0; c < cols; ++c) {
      queries[static_cast<size_t>(c) * width + i] = qv;
    }
    actual_k[i] = weight_type == GGML_TYPE_F16 ? rounded_f16(kv) : kv;
    // The F16 CPU type trait rounds the F32 query before its dot product.
    actual_q[i] = weight_type == GGML_TYPE_F16 ? rounded_f16(qv) : qv;
    const double product = actual_k[i] * actual_q[i];
    reference += product;
    absolute_sum += std::fabs(product);
  }
  double expected = reference;
  double tolerance = 1e-7 + 2e-6 * absolute_sum;
  if (cols == 1 && weight_type == GGML_TYPE_F16 && f16_sums) {
    const double blocked = blocked_f16_dot(actual_k, actual_q);
    expected = std::isfinite(blocked) ? static_cast<float>(blocked)
                                      : f32_recompute_dot(actual_k, actual_q);
    tolerance = 0.0;
  }
  if (weight_type == GGML_TYPE_F16) {
    ggml_backend_tensor_set(padded, half_keys.data(), 0, half_keys.size() * sizeof(ggml_fp16_t));
  } else {
    ggml_backend_tensor_set(padded, keys.data(), 0, keys.size() * sizeof(float));
  }
  ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
  graph.compute();
  std::vector<float> actual(static_cast<size_t>(rows) * cols);
  ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
  for (float value : actual) {
    if (!check_result(value, expected, tolerance, name, width, cols)) {
      break;
    }
  }
}

// Patch 0006 sums each F16 GEMM lane over at most this many K elements (eight
// lanes, so 64 products per lane) before adding it in F32.
constexpr int gemm_block = 512;
constexpr int gemm_block_steps = gemm_block / 8;

enum class Pattern {
  positive, cancellation, large_product, maximum, subnormal,
  // Every lane of every GEMM block sums exactly to 65504, the F16 maximum.
  block_maximum,
  // As block_maximum but each lane sums to 65536, which overflows in F16.
  block_overflow,
  // Each lane sums 64 products to 2047, exact in F16; a 65th product of 32
  // gives the odd 2079, which F16 cannot hold.
  block_exact,
  // block_exact for the single-column dot's 64 steps of 32 elements.
  dot_block_exact,
  // Lanes 0-7 of the first step take 2048, after which each product of 1
  // rounds away in F16; with the other accumulators' lanes this pins
  // upstream's F16 multiply-add, reduction order and block length.
  f16_rounding,
};

void dot_case(ggml_backend_t backend, int width, float magnitude, Pattern pattern,
              ggml_type weight_type, const char *name, int cols = 1, int padding = 8) {
  std::vector<float> keys(width), queries(width);
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
    } else if (pattern == Pattern::block_maximum || pattern == Pattern::block_overflow) {
      const bool last_step = (i % gemm_block) / 8 == gemm_block_steps - 1;
      kv = 32.0f;
      qv = 2048.0f / gemm_block_steps -
           (pattern == Pattern::block_maximum && last_step ? 1.0f : 0.0f);
    } else if (pattern == Pattern::block_exact) {
      kv = (i % gemm_block) / 8 == 1 ? 31.0f : 32.0f;
      qv = 1.0f;
    } else if (pattern == Pattern::dot_block_exact) {
      kv = (i % dot_block) / dot_step == 1 ? 31.0f : 32.0f;
      qv = 1.0f;
    } else if (pattern == Pattern::f16_rounding) {
      kv = i < 8 ? 64.0f : 1.0f;
      qv = i < 8 ? 32.0f : 1.0f;
    }
    keys[i] = kv;
    queries[i] = qv;
  }
  values_case(backend, keys, queries, weight_type, name, cols, padding);
}

// One F16 multiply-add of 256 * 256 = 65536 overflows the lane at `index`
// (step index / 32, position index % 32); a second index takes -65536, and
// `negative` makes the first -65536 as well. All other products are 1, so
// every exact result is finite.
void lane_overflow_case(ggml_backend_t backend, int width, int index, const char *name,
                        int negative_index = -1, bool negative = false) {
  std::vector<float> keys(width, 1.0f), queries(width, 1.0f);
  keys[index] = 256.0f;
  queries[index] = negative ? -256.0f : 256.0f;
  if (negative_index >= 0) {
    keys[negative_index] = 256.0f;
    queries[negative_index] = -256.0f;
  }
  values_case(backend, keys, queries, GGML_TYPE_F16, name);
}

// Products of mixed binary exponents with fractions that F32 sums round, so a
// changed order of the F32 additions changes the result. The lane `i % 8`
// takes the exponent. Each element past the last full step is 3 * 2^-11: under
// half an ulp of the row's sum (about 10^5 at width 4133), but five of them
// are not, so a tail summed in float differs from one summed in double.
// `overflow_index` >= 0 adds a product of 65536 there.
void spread_case(ggml_backend_t backend, int width, int overflow_index, const char *name) {
  constexpr int exponents[8] = {5, -7, 3, -9, 6, -6, 2, -10};
  const int np = width - width % dot_step;
  std::vector<float> keys(width), queries(width);
  for (int i = 0; i < width; ++i) {
    if (i >= np) {
      keys[i] = 0x3p-8f;
      queries[i] = 0x1p-3f;
      continue;
    }
    const int exponent = exponents[i % 8];
    keys[i] = (1.0f + ((7 * i + 3) % 11) / 16.0f) * std::ldexp(1.0f, exponent / 2);
    queries[i] = (1.0f + ((5 * i + 1) % 13) / 32.0f) * std::ldexp(1.0f, exponent - exponent / 2);
  }
  if (overflow_index >= 0) {
    keys[overflow_index] = queries[overflow_index] = 256.0f;
  }
  values_case(backend, keys, queries, GGML_TYPE_F16, name);
}

// A NaN or infinite input stays nonfinite, in the F16 steps and in the tail.
void nonfinite_input_case(ggml_backend_t backend, int width, int index, float value,
                          const char *name) {
  std::vector<float> keys(width, 1.0f), queries(width, 1.0f);
  keys[index] = value;
  values_case(backend, keys, queries, GGML_TYPE_F16, name);
}

// Sixteen weight rows times twelve query columns with exact integer products.
// In the K tail after one GEMM block, lane 0 overflows F16 for rows 5-6 times
// columns 8-9 and lane 1 for rows 1 and 7 times columns 2 and 11. The 4x6 tiles
// sit at rows 4i..4i+3 and columns 6j..6j+5, so tile (4-7, 0-5) overflows only
// in its last row (7, 2), tile (0-3, 6-11) only in its last column (1, 11), and
// the finiteness check must see every cell to redo each tile in F32.
void gemm_tile_case(ggml_backend_t backend) {
  constexpr int rows = 16;
  constexpr int cols = 12;
  constexpr int width = gemm_block + 8;
  constexpr int padding = 8;
  constexpr int stride = width + padding;
  Graph graph(backend);
  auto *padded = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, stride, rows);
  auto *k = ggml_view_2d(graph.ctx, padded, width, rows, padded->nb[1], 0);
  auto *q = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, width, cols);
  auto *out = ggml_mul_mat(graph.ctx, k, q);
  ggml_mul_mat_set_prec(out, GGML_PREC_F32);
  graph.allocate(out);

  std::vector<ggml_fp16_t> keys(static_cast<size_t>(stride) * rows, ggml_fp32_to_fp16(NAN));
  std::vector<float> queries(static_cast<size_t>(width) * cols);
  auto key = [](int r, int i) {
    const int lane = i - gemm_block;
    const bool large = (lane == 0 && (r == 5 || r == 6)) || (lane == 1 && (r == 1 || r == 7));
    return large ? 4096.0f : 1.0f + r % 3;
  };
  auto query = [](int c, int i) {
    const int lane = i - gemm_block;
    const bool large = (lane == 0 && (c == 8 || c == 9)) || (lane == 1 && (c == 2 || c == 11));
    return large ? 16.0f : i >= gemm_block ? 1.0f : 1.0f + c % 4;
  };
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < width; ++i) {
      keys[static_cast<size_t>(r) * stride + i] = ggml_fp32_to_fp16(key(r, i));
    }
  }
  for (int c = 0; c < cols; ++c) {
    for (int i = 0; i < width; ++i) {
      queries[static_cast<size_t>(c) * width + i] = query(c, i);
    }
  }
  ggml_backend_tensor_set(padded, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
  graph.compute();
  std::vector<float> actual(static_cast<size_t>(rows) * cols);
  ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < rows; ++r) {
      double expected = 0.0;
      for (int i = 0; i < width; ++i) {
        expected += static_cast<double>(key(r, i)) * query(c, i);
      }
      check_result(actual[static_cast<size_t>(c) * rows + r], expected, 0.0, "gemm-tile",
                   width, cols);
    }
  }
}

// Each key score is 128 * (128 + r / 8): the first half of its products sums
// to 2^21 and the second half cancels it, so F16 partial sums overflow while
// the softmax over the 16 keys stays far from saturated.
void attention_case(ggml_backend_t backend, bool flash, int queries_count) {
  constexpr int width = 256;
  constexpr int keys_count = 16;
  Graph graph(backend);
  auto *q = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F32, width, queries_count);
  auto *k = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16, width, keys_count);
  auto *v = ggml_new_tensor_2d(graph.ctx, GGML_TYPE_F16,
                               flash ? width : keys_count, flash ? keys_count : width);
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

  std::vector<float> queries(static_cast<size_t>(width) * queries_count, 128.0f);
  std::vector<ggml_fp16_t> keys(static_cast<size_t>(width) * keys_count);
  std::vector<ggml_fp16_t> value_data(static_cast<size_t>(width) * keys_count);
  std::vector<double> logits(keys_count);
  double max_logit = -INFINITY;
  for (int r = 0; r < keys_count; ++r) {
    double score = 0.0;
    for (int i = 0; i < width; ++i) {
      const float kv = i < width / 2 ? 128.0f : i < width - 1 ? -128.0f : 0.125f * r;
      keys[static_cast<size_t>(r) * width + i] = ggml_fp32_to_fp16(kv);
      score += static_cast<double>(kv) * 128.0;
    }
    logits[r] = score * scale;
    max_logit = std::max(max_logit, logits[r]);
  }
  std::vector<double> probabilities(keys_count);
  double denominator = 0.0;
  for (int r = 0; r < keys_count; ++r) {
    probabilities[r] = std::exp(logits[r] - max_logit);
    denominator += probabilities[r];
  }
  std::vector<double> expected(width, 0.0);
  for (int r = 0; r < keys_count; ++r) {
    probabilities[r] /= denominator;
    // Ordinary F16 V matmul also rounds its probability input to F16.
    const double p = flash ? probabilities[r] : rounded_f16(static_cast<float>(probabilities[r]));
    for (int i = 0; i < width; ++i) {
      const float value = 1.0f + ((i + r) % 7) * 0.125f;
      value_data[flash ? static_cast<size_t>(r) * width + i
                       : static_cast<size_t>(i) * keys_count + r] = ggml_fp32_to_fp16(value);
      expected[i] += value * p;
    }
  }
  ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
  ggml_backend_tensor_set(k, keys.data(), 0, keys.size() * sizeof(ggml_fp16_t));
  ggml_backend_tensor_set(v, value_data.data(), 0, value_data.size() * sizeof(ggml_fp16_t));
  graph.compute();
  std::vector<float> actual(static_cast<size_t>(width) * queries_count);
  ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
  // Flash attention accumulates V in F16 on one of its CPU paths.
  const double tolerance = flash ? 4e-3 : 1e-3;
  for (size_t j = 0; j < actual.size(); ++j) {
    if (!check_result(actual[j], expected[j % width], tolerance,
                      flash ? "flash-attention" : "attention", width, queries_count)) {
      break;
    }
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
  bool fp16_va = false;
  bool sve = false;
  for (auto *feature = get_features(reg); feature->name != nullptr; ++feature) {
    std::printf(" %s=%s", feature->name, feature->value);
    fp16_va |= std::strcmp(feature->name, "FP16_VA") == 0;
    sve |= std::strcmp(feature->name, "SVE") == 0;
  }
  f16_sums = fp16_va && !sve;
  std::printf("\nSingle-column F16 dot expectation: %s\n",
              f16_sums ? "upstream F16 sums in 64-step blocks" : "exact");

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
  for (int width : {33, 39, 40, 41, 63, 65, 127, 129}) {
    dot_case(backend, width, 0.0f, Pattern::maximum, GGML_TYPE_F16, "maximum-tail");
    dot_case(backend, width, 1.0f, Pattern::positive, GGML_TYPE_F16, "unpadded-tail", 1, 0);
  }
  for (int width : {dot_block - dot_step, dot_block, dot_block + dot_step, dot_block + 39,
                    2 * dot_block + dot_step + 7}) {
    dot_case(backend, width, 0.0f, Pattern::dot_block_exact, GGML_TYPE_F16, "dot-block-exact");
  }
  for (int width : {dot_block - 1, dot_block + dot_step - 1, 2 * dot_block + 5}) {
    dot_case(backend, width, 0.0f, Pattern::f16_rounding, GGML_TYPE_F16, "f16-rounding");
  }
  {
    // Rows of one block: acc0 lane 7 (high half) of the first step and acc3
    // lane 7 of the second; overflows of opposite sign in acc0 lanes 0 and 4,
    // and in acc0 and acc2 lane 0.
    constexpr int width = 2 * dot_step + 5;
    for (int index : {7, dot_step + 31}) {
      lane_overflow_case(backend, width, index, "short-lane-overflow");
    }
    lane_overflow_case(backend, width, 0, "short-split-overflow", dot_step + 4);
    lane_overflow_case(backend, width, 0, "short-split-overflow", dot_step + 16);
    for (int index : {7, dot_step + 31, width - 3}) {
      lane_overflow_case(backend, width, index, "short-negative-overflow", -1, true);
    }
  }
  {
    constexpr int width = 2 * dot_block + dot_step + 5;
    // The first block's acc0 lane 0, the second's acc0 lane 7 (high half),
    // the last step of the first block in acc2 lane 4, and the final partial
    // block's acc3 lane 7.
    for (int index : {3 * dot_step, (dot_block_steps + 5) * dot_step + 7,
                      (dot_block_steps - 1) * dot_step + 20, 2 * dot_block + 31}) {
      lane_overflow_case(backend, width, index, "lane-overflow");
    }
    // +inf in lane 0 and -inf in lane 4 meet in the F32 sum; +inf in acc0 and
    // -inf in acc2 meet in the F16 reduction. Both give NaN, not infinity.
    lane_overflow_case(backend, width, 3 * dot_step, "split-overflow-f32",
                       (dot_block_steps + 6) * dot_step + 4);
    lane_overflow_case(backend, width, 3 * dot_step, "split-overflow-f16",
                       4 * dot_step + 16);
    for (int index : {3 * dot_step, 2 * dot_block + 31}) {
      lane_overflow_case(backend, width, index, "negative-overflow", -1, true);
    }
  }
  // Elsewhere F32 sums of these rows drift past the exact-sum tolerance.
  if (f16_sums) {
    constexpr int short_width = 2 * dot_step + 5;
    constexpr int long_width = 2 * dot_block + dot_step + 5;
    spread_case(backend, 64, -1, "spread");
    spread_case(backend, short_width, -1, "spread-tail");
    spread_case(backend, short_width, 11, "spread-recompute");
    spread_case(backend, short_width + 3, dot_step + 20, "spread-recompute-tail");
    spread_case(backend, long_width, -1, "spread-blocks");
    spread_case(backend, long_width + 3, -1, "spread-blocks-tail");
    spread_case(backend, long_width, 5 * dot_step + 3, "spread-blocks-recompute");
  }
  // Upstream's other F16 dot paths assert a finite sum in debug builds.
  if (f16_sums) {
    for (int width : {2 * dot_step + 5, 2 * dot_block + dot_step + 5}) {
      for (int index : {9, width - 3}) {
        nonfinite_input_case(backend, width, index, NAN, "nan-input");
        nonfinite_input_case(backend, width, index, INFINITY, "inf-input");
      }
    }
  }
  dot_case(backend, 256, 128.0f, Pattern::cancellation, GGML_TYPE_F16, "hidden-overflow");
  for (int cols : {8, 16, 32}) {
    dot_case(backend, 64, 100.0f, Pattern::positive, GGML_TYPE_F16, "gemm-positive", cols);
    dot_case(backend, 256, 128.0f, Pattern::cancellation, GGML_TYPE_F16, "gemm-hidden-overflow", cols);
  }
  dot_case(backend, 32, 0.0f, Pattern::large_product, GGML_TYPE_F16, "gemm-large-product", 8);
  for (int width : {gemm_block, 2 * gemm_block + 8}) {
    dot_case(backend, width, 0.0f, Pattern::block_maximum, GGML_TYPE_F16,
             "gemm-block-maximum", 8);
    dot_case(backend, width, 0.0f, Pattern::block_overflow, GGML_TYPE_F16,
             "gemm-block-overflow", 8);
  }
  for (int width : {8, gemm_block - 8, gemm_block, gemm_block + 8, 4 * gemm_block + 8}) {
    dot_case(backend, width, 0.0f, Pattern::block_exact, GGML_TYPE_F16, "gemm-block-exact", 8);
  }
  gemm_tile_case(backend);
  dot_case(backend, 64, 100.0f, Pattern::positive, GGML_TYPE_F32, "gemm-f32-control", 8);
  for (int queries_count : {1, 8}) {
    attention_case(backend, false, queries_count);
    attention_case(backend, true, queries_count);
  }
  ggml_backend_free(backend);
#if defined(LLAMADART_TEST_CPU_BACKEND)
  ggml_backend_unload(reg);
#endif
  if (failures != 0) {
    std::fprintf(stderr, "FAIL: %d checks\n", failures);
    return 1;
  }
  std::printf("PASS: finite F16 dot and GEMM, tails, controls and ordinary/flash attention\n");
  return 0;
}
