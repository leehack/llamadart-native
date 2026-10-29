#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_cancel_eval_internal.h"
#include "llama_dart_wrapper.h"

#include "ggml-backend.h"
#include "ggml.h"
#include "mtmd.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

double elapsed_ms(clock_type::time_point from, clock_type::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

void init_backends() {
  llama_backend_init();
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
}

void test_abort_callback_reads_byte() {
  assert(!llama_dart_cancel_flag_abort_callback(nullptr));
  int8_t flag = 0;
  assert(!llama_dart_cancel_flag_abort_callback(&flag));
  flag = 1;
  assert(llama_dart_cancel_flag_abort_callback(&flag));
  flag = INT8_MIN;
  assert(llama_dart_cancel_flag_abort_callback(&flag));
  flag = 0;
  assert(!llama_dart_cancel_flag_abort_callback(&flag));
}

void test_eval_callback_requests_nothing_outside_a_call() {
  ggml_init_params params = {16 * 1024, nullptr, true};
  ggml_context *ctx = ggml_init(params);
  assert(ctx != nullptr);
  ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
  ggml_tensor *product = ggml_mul_mat(ctx, w, x);
  product->flags |= GGML_TENSOR_FLAG_COMPUTE;
  int8_t flag = 1;
  for (int i = 0; i < 4; ++i) {
    assert(!llama_dart_mtmd_eval_callback(product, true, nullptr));
    assert(!llama_dart_mtmd_eval_callback(product, true, &flag));
  }
  assert(llama_dart_mtmd_eval_callback(product, false, nullptr));
  assert(llama_dart_mtmd_eval_callback(product, false, &flag));
  ggml_free(ctx);
}

void test_cancellable_encode_does_not_start_when_cancelled() {
  int8_t flag = 1;
  llama_dart_clear_last_error();
  assert(llama_dart_mtmd_encode_chunk_cancellable(nullptr, nullptr, &flag) ==
         LLAMA_DART_MTMD_ENCODE_CANCELLED);
  assert(llama_dart_last_error() == nullptr);
}

void test_eval_answer_chunks_after_attention() {
  ggml_init_params params = {64 * 1024, nullptr, true};
  ggml_context *ctx = ggml_init(params);
  assert(ctx != nullptr);
  ggml_tensor *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
  ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 2);
  ggml_tensor *product = ggml_mul_mat(ctx, w, x);
  ggml_tensor *bias = ggml_add(ctx, product, product);
  ggml_tensor *attention = ggml_soft_max(ctx, product);
  ggml_tensor *idle_attention = ggml_soft_max(ctx, product);
  for (ggml_tensor *node : {product, bias, attention}) {
    node->flags |= GGML_TENSOR_FLAG_COMPUTE;
  }
  const double product_work = llama_dart_tts_eval_node_work(product);

  llama_dart_cancel_eval_state state(nullptr);
  state.budget = 2 * product_work;
  assert(!llama_dart_cancel_eval_answer(&state, attention, true));
  assert(!llama_dart_cancel_eval_answer(&state, product, true));
  assert(!llama_dart_cancel_eval_answer(&state, product, true));
  assert(state.work >= state.budget);
  assert(!llama_dart_cancel_eval_answer(&state, bias, true));
  assert(!llama_dart_cancel_eval_answer(&state, product, true));
  assert(!llama_dart_cancel_eval_answer(&state, idle_attention, true));
  assert(llama_dart_cancel_eval_answer(&state, attention, true));
  assert(state.work == 0.0);
  assert(!llama_dart_cancel_eval_answer(&state, attention, true));
  assert(llama_dart_cancel_eval_answer(&state, attention, false));
  assert(!state.observed.load());

  std::atomic<int8_t> flag{0};
  llama_dart_cancel_eval_state cancelling(
      reinterpret_cast<const int8_t *>(&flag));
  assert(!llama_dart_cancel_eval_answer(&cancelling, product, true));
  assert(llama_dart_cancel_eval_answer(&cancelling, product, false));
  flag.store(1);
  assert(!llama_dart_cancel_eval_answer(&cancelling, idle_attention, true));
  assert(llama_dart_cancel_eval_answer(&cancelling, bias, true));
  assert(!llama_dart_cancel_eval_answer(&cancelling, bias, false));
  flag.store(0);
  assert(llama_dart_cancel_eval_answer(&cancelling, product, true));
  assert(!llama_dart_cancel_eval_answer(&cancelling, product, false));
  assert(cancelling.observed.load());
  ggml_free(ctx);
}

int test_flag() {
  test_abort_callback_reads_byte();
  test_eval_callback_requests_nothing_outside_a_call();
  test_eval_answer_chunks_after_attention();
  test_cancellable_encode_does_not_start_when_cancelled();
  return 0;
}

// Context evaluation hook of the decode scenario: once armed, it has every
// node computed on its own and sets the cancel byte after node
// g_store_after_node.
std::atomic<int8_t> g_decode_flag{0};
std::atomic<bool> g_split_nodes{false};
int g_store_after_node = -1;
int g_nodes_done = 0;
int g_nodes_done_after_store = 0;

bool split_and_store(ggml_tensor *, bool ask, void *) {
  if (ask) {
    return g_split_nodes.load();
  }
  if (g_decode_flag.load() != 0) {
    ++g_nodes_done_after_store;
  }
  if (g_nodes_done++ == g_store_after_node) {
    g_decode_flag.store(1);
  }
  return true;
}

struct model_fixture {
  llama_model *model = nullptr;
  llama_context *context = nullptr;
  std::vector<llama_token> tokens;
};

model_fixture load_cpu_model(const char *path,
                             ggml_backend_sched_eval_callback eval_callback) {
  model_fixture fixture;
  auto model_params = llama_model_default_params();
  static ggml_backend_dev_t no_devices[] = {nullptr};
  model_params.n_gpu_layers = 0;
  model_params.devices = no_devices;
  fixture.model = llama_dart_model_load_from_file(path, model_params);
  assert(fixture.model != nullptr);
  auto context_params = llama_context_default_params();
  context_params.n_ctx = 256;
  context_params.offload_kqv = false;
  context_params.op_offload = false;
  context_params.cb_eval = eval_callback;
  fixture.context = llama_dart_init_from_model(fixture.model, context_params);
  assert(fixture.context != nullptr);
  for (llama_token token = 1; token <= 32; ++token) {
    fixture.tokens.push_back(token);
  }
  return fixture;
}

int32_t decode_prompt(model_fixture &fixture) {
  assert(llama_dart_memory_clear(llama_get_memory(fixture.context), true));
  return llama_dart_decode(
      fixture.context,
      llama_batch_get_one(fixture.tokens.data(),
                          static_cast<int32_t>(fixture.tokens.size())));
}

int test_decode(const char *path) {
  init_backends();
  model_fixture fixture = load_cpu_model(path, split_and_store);
  llama_set_abort_callback(
      fixture.context, llama_dart_cancel_flag_abort_callback, &g_decode_flag);
  llama_memory_t memory = llama_get_memory(fixture.context);

  assert(decode_prompt(fixture) == 0);
  assert(llama_memory_seq_pos_max(memory, 0) == 31);

  g_decode_flag.store(1);
  assert(decode_prompt(fixture) == 2);
  assert(llama_memory_seq_pos_max(memory, 0) == -1);

  g_decode_flag.store(0);
  assert(decode_prompt(fixture) == 0);
  assert(llama_memory_seq_pos_max(memory, 0) == 31);

  g_split_nodes.store(true);
  g_store_after_node = 3;
  assert(decode_prompt(fixture) == 2);
  assert(g_nodes_done == 4);
  assert(g_nodes_done_after_store == 0);
  assert(llama_memory_seq_pos_max(memory, 0) == -1);

  g_split_nodes.store(false);
  g_decode_flag.store(0);
  assert(decode_prompt(fixture) == 0);

  llama_set_abort_callback(fixture.context, nullptr, nullptr);
  g_decode_flag.store(1);
  assert(decode_prompt(fixture) == 0);

  llama_dart_exit_free(fixture.context);
  llama_dart_exit_free(fixture.model);
  return 0;
}

struct encode_fixture {
  llama_model *model = nullptr;
  mtmd_context *mtmd = nullptr;
  mtmd_input_chunks *chunks = nullptr;
  const mtmd_input_chunk *chunk = nullptr;
  size_t embd_floats = 0;
};

encode_fixture load_encoder(const char *model_path, const char *mmproj_path,
                            bool use_gpu) {
  encode_fixture fixture;
  auto model_params = llama_model_default_params();
  static ggml_backend_dev_t no_devices[] = {nullptr};
  model_params.n_gpu_layers = use_gpu ? 99 : 0;
  model_params.devices = use_gpu ? nullptr : no_devices;
  fixture.model = llama_dart_model_load_from_file(model_path, model_params);
  assert(fixture.model != nullptr);

  mtmd_context_params params = mtmd_context_params_default();
  params.use_gpu = use_gpu;
  params.print_timings = false;
  params.warmup = false;
  params.cb_eval = llama_dart_mtmd_eval_callback;
  fixture.mtmd =
      llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
  assert(fixture.mtmd != nullptr);

  mtmd_bitmap *bitmap = nullptr;
  if (mtmd_support_vision(fixture.mtmd)) {
    const uint32_t side = 768;
    std::vector<unsigned char> rgb(side * side * 3);
    for (size_t i = 0; i < rgb.size(); ++i) {
      rgb[i] = static_cast<unsigned char>((i * 2654435761u) >> 24);
    }
    bitmap = mtmd_bitmap_init(side, side, rgb.data());
  } else {
    assert(mtmd_support_audio(fixture.mtmd));
    std::vector<float> pcm(16000 * 20);
    for (size_t i = 0; i < pcm.size(); ++i) {
      pcm[i] = 0.25f * std::sin(static_cast<float>(i) * 0.05f);
    }
    bitmap = llama_dart_mtmd_bitmap_init_from_audio(pcm.size(), pcm.data());
  }
  assert(bitmap != nullptr);

  fixture.chunks = mtmd_input_chunks_init();
  const char *marker = mtmd_default_marker();
  mtmd_input_text text = {marker, std::strlen(marker), true, true};
  const mtmd_bitmap *bitmaps[] = {bitmap};
  assert(llama_dart_mtmd_tokenize(fixture.mtmd, fixture.chunks, &text, bitmaps,
                                  1) == 0);
  mtmd_bitmap_free(bitmap);
  for (size_t i = 0; i < mtmd_input_chunks_size(fixture.chunks); ++i) {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(fixture.chunks, i);
    if (mtmd_input_chunk_get_type(chunk) != MTMD_INPUT_CHUNK_TYPE_TEXT) {
      fixture.chunk = chunk;
      break;
    }
  }
  assert(fixture.chunk != nullptr);
  fixture.embd_floats =
      static_cast<size_t>(llama_model_n_embd_inp(fixture.model)) *
      mtmd_input_chunk_get_n_tokens(fixture.chunk);
  return fixture;
}

std::vector<float> output(const encode_fixture &fixture) {
  const float *embd = mtmd_get_output_embd(fixture.mtmd);
  return std::vector<float>(embd, embd + fixture.embd_floats);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

// Largest difference from expected, as a fraction of expected's largest
// magnitude.
double relative_difference(const std::vector<float> &actual,
                           const std::vector<float> &expected) {
  double difference = 0.0;
  double magnitude = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    difference = std::max(difference,
                          std::fabs(double(actual[i]) - double(expected[i])));
    magnitude = std::max(magnitude, std::fabs(double(expected[i])));
  }
  return magnitude > 0.0 ? difference / magnitude : difference;
}

// Needs a real model and projector, so it runs only when the build is given
// them. It prints the cost of the cancellable encode and the cancel latency.
int test_mtmd_encode(const char *model_path, const char *mmproj_path,
                     bool use_gpu) {
  init_backends();
  encode_fixture fixture = load_encoder(model_path, mmproj_path, use_gpu);
  std::atomic<int8_t> flag{0};
  const int8_t *flag_byte = reinterpret_cast<const int8_t *>(&flag);
  constexpr int kRuns = 5;

  assert(llama_dart_mtmd_encode_chunk(fixture.mtmd, fixture.chunk) == 0);
  const std::vector<float> plain = output(fixture);
  assert(llama_dart_mtmd_encode_chunk_cancellable(fixture.mtmd, fixture.chunk,
                                                  flag_byte) == 0);
  const std::vector<float> chunked = output(fixture);
  // The CPU backend fuses nothing, so chunking leaves its output unchanged.
  // Metal fuses only within one command buffer, whose node ranges depend on
  // the size of the graph it computes, so there chunking changes which ops
  // are fused.
  const double chunked_difference = relative_difference(chunked, plain);
  assert(use_gpu ? chunked_difference < 0.05 : chunked == plain);

  std::vector<double> plain_ms;
  std::vector<double> cancellable_ms;
  for (int i = 0; i < kRuns; ++i) {
    auto start = clock_type::now();
    assert(llama_dart_mtmd_encode_chunk(fixture.mtmd, fixture.chunk) == 0);
    plain_ms.push_back(elapsed_ms(start, clock_type::now()));
    assert(output(fixture) == plain);

    start = clock_type::now();
    assert(llama_dart_mtmd_encode_chunk_cancellable(fixture.mtmd, fixture.chunk,
                                                    flag_byte) == 0);
    cancellable_ms.push_back(elapsed_ms(start, clock_type::now()));
    assert(output(fixture) == chunked);
  }
  const double full_ms = median(plain_ms);

  // Without a cancellable call in flight the callback stops nothing.
  flag.store(1);
  assert(llama_dart_mtmd_encode_chunk(fixture.mtmd, fixture.chunk) == 0);
  assert(output(fixture) == plain);
  assert(llama_dart_mtmd_encode_chunk_cancellable(fixture.mtmd, fixture.chunk,
                                                  flag_byte) ==
         LLAMA_DART_MTMD_ENCODE_CANCELLED);

  // A store after the last chunk boundary lets the encode finish, and a
  // loaded machine can wake the canceller late, so not every run cancels.
  std::vector<double> latency_ms;
  for (int i = 0; i < 4 * kRuns && latency_ms.size() < kRuns; ++i) {
    flag.store(0);
    const auto delay = std::chrono::duration<double, std::milli>(
        full_ms * (0.2 + 0.1 * (i % kRuns)));
    std::atomic<clock_type::rep> stored{0};
    const auto start = clock_type::now();
    std::thread canceller([&] {
      std::this_thread::sleep_until(
          start + std::chrono::duration_cast<clock_type::duration>(delay));
      stored.store(clock_type::now().time_since_epoch().count());
      flag.store(1);
    });
    const int32_t status = llama_dart_mtmd_encode_chunk_cancellable(
        fixture.mtmd, fixture.chunk, flag_byte);
    const auto returned = clock_type::now();
    canceller.join();
    const clock_type::time_point stored_at{clock_type::duration{stored.load()}};
    if (status == 0) {
      assert(output(fixture) == chunked);
      continue;
    }
    assert(status == LLAMA_DART_MTMD_ENCODE_CANCELLED);
    assert(returned > stored_at);
    latency_ms.push_back(elapsed_ms(stored_at, returned));
  }
  assert(latency_ms.size() == kRuns);

  flag.store(0);
  assert(llama_dart_mtmd_encode_chunk_cancellable(fixture.mtmd, fixture.chunk,
                                                  flag_byte) == 0);
  assert(output(fixture) == chunked);

  printf("mtmd-encode %s, %zu output floats: plain %.2f ms, cancellable "
         "%.2f ms (median of %d), chunked output differs by %.3g of its "
         "largest magnitude; cancel store to return %.2f-%.2f ms, median "
         "%.2f\n",
         use_gpu ? "gpu" : "cpu", fixture.embd_floats, full_ms,
         median(cancellable_ms), kRuns, chunked_difference,
         *std::min_element(latency_ms.begin(), latency_ms.end()),
         *std::max_element(latency_ms.begin(), latency_ms.end()),
         median(latency_ms));

  mtmd_input_chunks_free(fixture.chunks);
  llama_dart_exit_free(fixture.mtmd);
  llama_dart_exit_free(fixture.model);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::string scenario = argc > 1 ? argv[1] : "flag";
  if (scenario == "flag") {
    return test_flag();
  }
  if (scenario == "decode" && argc == 3) {
    return test_decode(argv[2]);
  }
  if (scenario == "mtmd-encode" && argc >= 4) {
    const bool use_gpu = argc > 4 && std::string(argv[4]) == "gpu";
    return test_mtmd_encode(argv[2], argv[3], use_gpu);
  }
  fprintf(stderr, "unknown scenario %s\n", scenario.c_str());
  return 1;
}
