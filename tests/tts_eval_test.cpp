#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_eval_internal.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t kDim = 64;
constexpr int64_t kCols = 8;
constexpr int kLayers = 6;
constexpr double kMatMulWork = static_cast<double>(kDim * kCols * kDim);
constexpr double kRowWork = static_cast<double>(kDim * kCols);

ggml_tensor *computed(ggml_tensor *tensor) {
  tensor->flags |= GGML_TENSOR_FLAG_COMPUTE;
  return tensor;
}

void test_node_work(ggml_context *ctx) {
  ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, 32);
  ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, kCols);
  ggml_tensor *product = ggml_mul_mat(ctx, a, b);
  assert(llama_dart_tts_eval_node_work(product) == 32.0 * kCols * kDim);
  assert(llama_dart_tts_eval_node_work(ggml_add(ctx, product, product)) ==
         32.0 * kCols);
  assert(llama_dart_tts_eval_node_work(ggml_view_1d(ctx, product, 8, 0)) ==
         0.0);
  assert(llama_dart_tts_eval_node_work(ggml_reshape_1d(ctx, product, 256)) ==
         0.0);
  assert(llama_dart_tts_eval_node_work(ggml_transpose(ctx, product)) == 0.0);
  assert(llama_dart_tts_eval_node_work(
             ggml_permute(ctx, product, 1, 0, 2, 3)) == 0.0);
  assert(llama_dart_tts_eval_node_work(a) == 0.0);
}

void test_boundaries_follow_budget_and_mul_mat(ggml_context *ctx) {
  ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, kDim);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, kCols);
  ggml_tensor *product = computed(ggml_mul_mat(ctx, w, x));
  ggml_tensor *sum = computed(ggml_add(ctx, product, product));
  ggml_tensor *idle_product = ggml_mul_mat(ctx, w, x);

  llama_dart_tts_eval_chunker chunker;
  chunker.budget = kMatMulWork + kRowWork;
  assert(!llama_dart_tts_eval_answer(&chunker, product, true, false));
  assert(!llama_dart_tts_eval_answer(&chunker, idle_product, true, false));
  assert(chunker.work == kMatMulWork);
  assert(!llama_dart_tts_eval_answer(&chunker, sum, true, false));
  assert(chunker.boundary_due);
  assert(!llama_dart_tts_eval_answer(&chunker, sum, true, false));
  assert(!llama_dart_tts_eval_answer(&chunker, idle_product, true, false));
  assert(llama_dart_tts_eval_answer(&chunker, product, true, false));
  assert(chunker.work == 0.0 && !chunker.boundary_due);
  assert(chunker.past_first_boundary);
  assert(!llama_dart_tts_eval_answer(&chunker, product, true, false));
  assert(llama_dart_tts_eval_answer(&chunker, product, false, false));
  assert(!llama_dart_tts_eval_answer(&chunker, product, false, true));
}

void test_cancel_waits_for_first_boundary(ggml_context *ctx) {
  ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, kDim);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kDim, kCols);
  ggml_tensor *product = computed(ggml_mul_mat(ctx, w, x));
  ggml_tensor *sum = computed(ggml_add(ctx, product, product));
  ggml_tensor *idle_sum = ggml_add(ctx, product, product);

  llama_dart_tts_eval_chunker chunker;
  chunker.budget = 2 * kMatMulWork;
  assert(!llama_dart_tts_eval_answer(&chunker, product, true, true));
  assert(!llama_dart_tts_eval_answer(&chunker, sum, true, true));
  assert(llama_dart_tts_eval_answer(&chunker, product, true, true));
  assert(llama_dart_tts_eval_answer(&chunker, sum, true, true));
  assert(!llama_dart_tts_eval_answer(&chunker, idle_sum, true, true));
  assert(!llama_dart_tts_eval_answer(&chunker, sum, true, false));
}

void test_cancel_flag_latches() {
  std::atomic<bool> latched{false};
  std::atomic<int8_t> flag{0};
  const int8_t *byte = reinterpret_cast<const int8_t *>(&flag);
  assert(!llama_dart_tts_cancel_observed(&latched, nullptr));
  assert(!llama_dart_tts_cancel_observed(&latched, byte));
  assert(!latched.load());
  flag.store(1);
  assert(llama_dart_tts_cancel_observed(&latched, byte));
  flag.store(0);
  assert(llama_dart_tts_cancel_observed(&latched, byte));
  assert(llama_dart_tts_cancel_observed(&latched, nullptr));

  std::atomic<bool> requested{true};
  assert(llama_dart_tts_cancel_observed(&requested, nullptr));
  assert(llama_dart_tts_cancel_observed(&requested, byte));
}

struct probe {
  llama_dart_tts_eval_chunker chunker;
  bool cancelled = false;
  bool cancel_at_first_boundary = false;
  bool stopped = false;
  int asks = 0;
  int asks_after_stop = 0;
  int boundaries = 0;
  int non_mul_mat_boundaries = 0;
};

bool probe_callback(ggml_tensor *node, bool ask, void *user_data) {
  auto *p = static_cast<probe *>(user_data);
  if (ask && p->stopped) {
    ++p->asks_after_stop;
  }
  const bool answer =
      llama_dart_tts_eval_answer(&p->chunker, node, ask, p->cancelled);
  if (!ask) {
    p->stopped = p->stopped || !answer;
    return answer;
  }
  ++p->asks;
  if (answer) {
    ++p->boundaries;
    p->non_mul_mat_boundaries += node->op == GGML_OP_MUL_MAT ? 0 : 1;
    p->cancelled = p->cancelled || p->cancel_at_first_boundary;
  }
  return answer;
}

struct scheduled_graph {
  ggml_backend_t backend = nullptr;
  ggml_context *weights_ctx = nullptr;
  ggml_backend_buffer_t weights = nullptr;
  ggml_context *graph_ctx = nullptr;
  ggml_cgraph *graph = nullptr;
  ggml_tensor *output = nullptr;
  ggml_backend_sched_t sched = nullptr;

  scheduled_graph() {
    ggml_backend_load_all();
    backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    assert(backend != nullptr);

    ggml_init_params weight_params = {
        ggml_tensor_overhead() * (2 * kLayers + 1), nullptr, true};
    weights_ctx = ggml_init(weight_params);
    ggml_tensor *input =
        ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, kDim, kCols);
    std::vector<ggml_tensor *> matrices;
    std::vector<ggml_tensor *> gains;
    for (int layer = 0; layer < kLayers; ++layer) {
      matrices.push_back(
          ggml_new_tensor_2d(weights_ctx, GGML_TYPE_F32, kDim, kDim));
      gains.push_back(ggml_new_tensor_1d(weights_ctx, GGML_TYPE_F32, kDim));
    }
    weights = ggml_backend_alloc_ctx_tensors(weights_ctx, backend);
    assert(weights != nullptr);
    fill(input, 1);
    for (int layer = 0; layer < kLayers; ++layer) {
      fill(matrices[layer], 7 + layer);
      fill(gains[layer], 101 + layer);
    }

    ggml_init_params graph_params = {
        ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE +
            ggml_graph_overhead(),
        nullptr, true};
    graph_ctx = ggml_init(graph_params);
    ggml_tensor *hidden = input;
    for (int layer = 0; layer < kLayers; ++layer) {
      hidden = ggml_mul_mat(graph_ctx, matrices[layer], hidden);
      hidden = ggml_rms_norm(graph_ctx, hidden, 1e-6f);
      hidden = ggml_mul(graph_ctx, hidden, gains[layer]);
    }
    output = hidden;
    ggml_set_output(output);
    graph = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, output);
    assert(ggml_graph_n_nodes(graph) == 3 * kLayers);

    sched = ggml_backend_sched_new(&backend, nullptr, 1,
                                   GGML_DEFAULT_GRAPH_SIZE, false, false);
    assert(sched != nullptr);
  }

  ~scheduled_graph() {
    ggml_backend_sched_free(sched);
    ggml_free(graph_ctx);
    ggml_backend_buffer_free(weights);
    ggml_free(weights_ctx);
    ggml_backend_free(backend);
  }

  static void fill(ggml_tensor *tensor, int seed) {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
    uint32_t state = static_cast<uint32_t>(seed) * 2654435761u;
    for (float &value : values) {
      state = state * 1664525u + 1013904223u;
      value = static_cast<float>(state >> 8) / 16777216.0f - 0.5f;
    }
    ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
  }

  ggml_status compute(probe *p) {
    ggml_backend_sched_reset(sched);
    ggml_backend_sched_set_eval_callback(sched, p ? probe_callback : nullptr,
                                         p);
    return ggml_backend_sched_graph_compute(sched, graph);
  }

  std::vector<float> read_output() const {
    std::vector<float> values(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    return values;
  }
};

// The budget first overflows on the first RMS_NORM, so the first boundary
// moves to the second MUL_MAT; every later MUL_MAT then ends a chunk.
constexpr double kSchedulerBudget = kMatMulWork + 0.5 * kRowWork;

void test_chunked_compute_matches_unchunked(scheduled_graph &g) {
  assert(g.compute(nullptr) == GGML_STATUS_SUCCESS);
  const std::vector<float> expected = g.read_output();

  probe p;
  p.chunker.budget = kSchedulerBudget;
  assert(g.compute(&p) == GGML_STATUS_SUCCESS);
  assert(p.boundaries == kLayers - 1);
  assert(p.non_mul_mat_boundaries == 0);
  assert(!p.stopped);
  const std::vector<float> chunked = g.read_output();
  assert(std::memcmp(expected.data(), chunked.data(),
                     expected.size() * sizeof(float)) == 0);
}

void test_cancel_stops_at_chunk_boundary(scheduled_graph &g) {
  probe p;
  p.chunker.budget = kSchedulerBudget;
  p.cancel_at_first_boundary = true;
  assert(g.compute(&p) == GGML_STATUS_SUCCESS);
  assert(p.stopped);
  assert(p.boundaries == 1);
  assert(p.asks == 4);
  assert(p.asks_after_stop == 0);
}

void test_early_cancel_runs_to_first_boundary(scheduled_graph &g) {
  probe p;
  p.chunker.budget = kSchedulerBudget;
  p.cancelled = true;
  assert(g.compute(&p) == GGML_STATUS_SUCCESS);
  assert(p.stopped);
  assert(p.boundaries == 1);
  assert(p.asks == 4);
  assert(p.asks_after_stop == 0);
}

} // namespace

int main() {
  ggml_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

  ggml_init_params params = {ggml_tensor_overhead() * 64, nullptr, true};
  ggml_context *ctx = ggml_init(params);
  test_node_work(ctx);
  test_boundaries_follow_budget_and_mul_mat(ctx);
  test_cancel_waits_for_first_boundary(ctx);
  test_cancel_flag_latches();
  ggml_free(ctx);

  scheduled_graph graph;
  test_chunked_compute_matches_unchunked(graph);
  test_cancel_stops_at_chunk_boundary(graph);
  test_early_cancel_runs_to_first_boundary(graph);
  return 0;
}
