#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_wrapper.h"

#include "gguf.h"
#include "mtmd.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LLAMADART_TEST_ADDRESS_SANITIZER 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__) && !defined(LLAMADART_TEST_ADDRESS_SANITIZER)
#define LLAMADART_TEST_ADDRESS_SANITIZER 1
#endif

#if defined(LLAMADART_TEST_ADDRESS_SANITIZER)
#include <sanitizer/asan_interface.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

// Each scenario runs in its own process: teardown runs once per process and
// leaves the registry unusable from other threads.

namespace {

std::mutex g_log_mutex;
std::vector<std::string> g_log;

void record(const std::string &event) {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  g_log.push_back(event);
}

std::vector<std::string> recorded() {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  return g_log;
}

void free_named(void *object) { record(static_cast<const char *>(object)); }

char *name(const char *value) { return const_cast<char *>(value); }

void sleep_ms(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Whether the heap block that began at object has been freed, for a build
// that can tell: AddressSanitizer poisons a freed block, and the macOS
// allocator reports no size for one until it hands the block out again.
#if defined(LLAMADART_TEST_ADDRESS_SANITIZER)
const bool kSeesFreedBlocks = true;
bool is_freed(const void *object) {
  return __asan_address_is_poisoned(object) != 0;
}
#elif defined(__APPLE__)
const bool kSeesFreedBlocks = true;
bool is_freed(const void *object) { return malloc_size(object) == 0; }
#else
const bool kSeesFreedBlocks = false;
bool is_freed(const void *) { return false; }
#endif

int64_t elapsed_ms(std::chrono::steady_clock::time_point since) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - since)
      .count();
}

char kSessionLate[] = "session-late";

const std::vector<std::string> kStageOrder = {
    "session-late", "session-early", "scheduler",
    "context-late", "context-early", "mtmd",
    "backend",      "model-late",    "model-early",
};

void track_out_of_stage_order() {
  const auto track = [](const char *value, int32_t stage) {
    assert(llama_dart_exit_track(name(value), free_named, stage));
  };
  track("model-early", LLAMA_DART_EXIT_STAGE_MODEL);
  track("backend", LLAMA_DART_EXIT_STAGE_BACKEND);
  track("context-early", LLAMA_DART_EXIT_STAGE_CONTEXT);
  track("session-early", LLAMA_DART_EXIT_STAGE_SESSION);
  track("mtmd", LLAMA_DART_EXIT_STAGE_MODEL_USER);
  track("model-late", LLAMA_DART_EXIT_STAGE_MODEL);
  track("scheduler", LLAMA_DART_EXIT_STAGE_SCHEDULER);
  track("context-late", LLAMA_DART_EXIT_STAGE_CONTEXT);
  track(kSessionLate, LLAMA_DART_EXIT_STAGE_SESSION);
}

int test_dispose() {
  assert(!llama_dart_exit_track(nullptr, free_named, 0));
  assert(!llama_dart_exit_track(name("a"), nullptr, 0));
  assert(!llama_dart_exit_track(name("a"), free_named, -1));
  assert(!llama_dart_exit_track(name("a"), free_named,
                                LLAMA_DART_EXIT_STAGE_MODEL + 1));
  assert(llama_dart_exit_tracked_count() == 0);

  char *model = name("model");
  char *context = name("context");
  char *kept = name("kept");
  assert(llama_dart_exit_track(model, free_named, LLAMA_DART_EXIT_STAGE_MODEL));
  assert(llama_dart_exit_track(context, free_named,
                               LLAMA_DART_EXIT_STAGE_CONTEXT));
  assert(llama_dart_exit_track(kept, free_named, LLAMA_DART_EXIT_STAGE_MODEL));
  assert(llama_dart_exit_tracked_count() == 3);

  llama_dart_exit_free(context);
  llama_dart_exit_free(model);
  llama_dart_exit_free(model);
  llama_dart_exit_free(nullptr);
  llama_dart_exit_free(name("never-tracked"));
  assert(recorded() == std::vector<std::string>({"context", "model"}));

  assert(llama_dart_exit_untrack(kept));
  assert(!llama_dart_exit_untrack(kept));
  llama_dart_exit_free(kept);
  assert(llama_dart_exit_tracked_count() == 0);

  llama_dart_exit_teardown();
  assert(recorded() == std::vector<std::string>({"context", "model"}));
  return 0;
}

int test_order() {
  track_out_of_stage_order();
  llama_dart_ngram *ngram = llama_dart_ngram_simple_init(4, 8);
  assert(ngram != nullptr);
  assert(llama_dart_exit_tracked_count() ==
         static_cast<int32_t>(kStageOrder.size()) + 1);

  llama_dart_exit_teardown();
  assert(recorded() == kStageOrder);
  assert(llama_dart_exit_tracked_count() == 0);

  // The teardown thread is not blocked, and nothing is freed a second time.
  llama_dart_exit_free(name("model-early"));
  assert(!llama_dart_exit_untrack(name("model-early")));
  assert(!llama_dart_exit_track(name("late"), free_named,
                                LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_exit_teardown();
  assert(recorded() == kStageOrder);
  return 0;
}

int test_repeat() {
  char *object = name("object");
  for (int i = 0; i < 2000; ++i) {
    assert(llama_dart_exit_track(object, free_named,
                                 LLAMA_DART_EXIT_STAGE_CONTEXT));
    llama_dart_ngram *ngram = llama_dart_ngram_simple_init(4, 8);
    assert(ngram != nullptr);
    assert(llama_dart_exit_tracked_count() == 2);
    llama_dart_ngram_free(ngram);
    llama_dart_exit_free(object);
    assert(llama_dart_exit_tracked_count() == 0);
  }
  assert(recorded().size() == 2000);

  // A tracked address that is tracked again holds a new object.
  assert(
      llama_dart_exit_track(object, free_named, LLAMA_DART_EXIT_STAGE_CONTEXT));
  assert(
      llama_dart_exit_track(object, free_named, LLAMA_DART_EXIT_STAGE_MODEL));
  assert(llama_dart_exit_tracked_count() == 1);
  llama_dart_exit_teardown();
  assert(recorded().size() == 2001);
  return 0;
}

int test_in_flight() {
  track_out_of_stage_order();
  llama_dart_exit_set_wait_ms(30000);

  std::atomic<bool> in_call{false};
  std::atomic<bool> returned{false};
  std::thread([&in_call, &returned] {
    llama_dart_exit_call_begin();
    in_call.store(true);
    sleep_ms(400);
    record("call-finished");
    llama_dart_exit_call_end();
    returned.store(true);
  }).detach();
  while (!in_call.load()) {
    sleep_ms(1);
  }

  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(elapsed_ms(started) < 20000);

  std::vector<std::string> expected = {"call-finished"};
  expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
  assert(recorded() == expected);

  // The thread that was in the call never gets back to its caller.
  sleep_ms(200);
  assert(!returned.load());
  return 0;
}

// Teardown waits for the outermost call of a thread, which stays free to make
// nested calls and to use the registry until that call ends.
int test_nested() {
  track_out_of_stage_order();
  llama_dart_exit_set_wait_ms(30000);
  static char inner[] = "inner";

  std::atomic<bool> in_call{false};
  std::atomic<bool> returned{false};
  std::thread([&in_call, &returned] {
    llama_dart_exit_call_begin();
    in_call.store(true);
    sleep_ms(300);
    llama_dart_exit_call_begin();
    assert(llama_dart_exit_track(inner, free_named,
                                 LLAMA_DART_EXIT_STAGE_SESSION));
    llama_dart_ngram *ngram = llama_dart_ngram_simple_init(4, 8);
    assert(ngram != nullptr);
    assert(llama_dart_ngram_begin(ngram, 0, nullptr, 0));
    llama_dart_ngram_free(ngram);
    llama_dart_exit_free(kSessionLate);
    llama_dart_exit_call_end();
    record("call-finished");
    llama_dart_exit_call_end();
    returned.store(true);
  }).detach();
  while (!in_call.load()) {
    sleep_ms(1);
  }

  llama_dart_exit_teardown();
  std::vector<std::string> expected = {"session-late", "call-finished",
                                       "inner"};
  expected.insert(expected.end(), kStageOrder.begin() + 1, kStageOrder.end());
  assert(recorded() == expected);
  sleep_ms(200);
  assert(!returned.load());
  return 0;
}

// A thread that left a call in flight gets time to finish what follows it.
int test_settle() {
  track_out_of_stage_order();

  std::atomic<bool> left_call{false};
  std::thread([&left_call] {
    llama_dart_exit_call_begin();
    llama_dart_exit_call_end();
    left_call.store(true);
    sleep_ms(50);
    record("tail-finished");
  }).detach();
  while (!left_call.load()) {
    sleep_ms(1);
  }

  llama_dart_exit_teardown();
  std::vector<std::string> expected = {"tail-finished"};
  expected.insert(expected.end(), kStageOrder.begin(), kStageOrder.end());
  assert(recorded() == expected);
  return 0;
}

int test_timeout() {
  track_out_of_stage_order();
  llama_dart_exit_set_wait_ms(1000);

  std::atomic<bool> in_call{false};
  std::thread([&in_call] {
    llama_dart_exit_call_begin();
    in_call.store(true);
    sleep_ms(600000);
  }).detach();
  while (!in_call.load()) {
    sleep_ms(1);
  }

  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  const int64_t waited = elapsed_ms(started);
  assert(waited >= 900);
  assert(waited < 20000);
  assert(recorded().empty());
  assert(llama_dart_exit_tracked_count() ==
         static_cast<int32_t>(kStageOrder.size()));

  // Teardown runs once: a second run neither waits nor frees.
  const auto restarted = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(elapsed_ms(restarted) < 500);
  assert(recorded().empty());
  return 0;
}

// A call that never ends does not hold up an exit that has nothing to free,
// and neither does a free that has finished.
int test_idle_wait() {
  char *object = name("object");
  assert(
      llama_dart_exit_track(object, free_named, LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_exit_free(object);
  std::thread([] { llama_dart_exit_call_begin(); }).join();
  llama_dart_exit_set_wait_ms(30000);
  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(elapsed_ms(started) < 5000);
  return 0;
}

// Teardown that runs inside a call in flight, as when a callback of that call
// exits the process, does not wait for the call it is in.
int test_own_call() {
  track_out_of_stage_order();
  llama_dart_exit_set_wait_ms(30000);
  llama_dart_exit_call_begin();
  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(elapsed_ms(started) < 5000);
  assert(recorded() == kStageOrder);
  llama_dart_exit_call_end();
  return 0;
}

int test_blocked() {
  char *object = name("object");
  assert(
      llama_dart_exit_track(object, free_named, LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_ngram *ngram = llama_dart_ngram_simple_init(4, 8);
  assert(ngram != nullptr);
  llama_dart_exit_teardown();
  assert(recorded() == std::vector<std::string>({"object"}));

  static std::atomic<int> started{0};
  static std::atomic<int> returned{0};
  int launched = 0;
  const auto late = [&launched](void (*call)(void *), void *argument) {
    ++launched;
    std::thread([call, argument] {
      started.fetch_add(1);
      call(argument);
      returned.fetch_add(1);
    }).detach();
  };
  late([](void *argument) { llama_dart_exit_free(argument); }, object);
  late([](void *argument) { llama_dart_exit_untrack(argument); }, object);
  late(
      [](void *argument) {
        llama_dart_exit_track(argument, free_named,
                              LLAMA_DART_EXIT_STAGE_MODEL);
      },
      object);
  late([](void *) { llama_dart_exit_call_begin(); }, nullptr);
  late([](void *) { llama_dart_ngram_simple_init(4, 8); }, nullptr);
  late(
      [](void *argument) {
        llama_dart_ngram_free(static_cast<llama_dart_ngram *>(argument));
      },
      ngram);
  late(
      [](void *argument) {
        llama_dart_ngram_begin(static_cast<llama_dart_ngram *>(argument), 0,
                               nullptr, 0);
      },
      ngram);
  late(
      [](void *) {
        llama_dart_model_load_from_file("missing.gguf",
                                        llama_model_default_params());
      },
      nullptr);
  late(
      [](void *) {
        llama_dart_init_from_model(nullptr, llama_context_default_params());
      },
      nullptr);
  late([](void *) { llama_dart_decode(nullptr, llama_batch{}); }, nullptr);
  late([](void *) { llama_dart_encode(nullptr, llama_batch{}); }, nullptr);
  late([](void *) { llama_dart_synchronize(nullptr); }, nullptr);
  late([](void *) { llama_dart_sampler_sample(nullptr, nullptr, 0); }, nullptr);
  late([](void *) { llama_dart_state_save_file(nullptr, "", nullptr, 0); },
       nullptr);
  late(
      [](void *) {
        llama_dart_state_load_file(nullptr, "", nullptr, 0, nullptr);
      },
      nullptr);
  late([](void *) { llama_dart_state_seq_get_size_ext(nullptr, 0, 0); },
       nullptr);
  late(
      [](void *) {
        llama_dart_state_seq_get_data_ext(nullptr, nullptr, 0, 0, 0);
      },
      nullptr);
  late(
      [](void *) {
        llama_dart_state_seq_set_data_ext(nullptr, nullptr, 0, 0, 0);
      },
      nullptr);
  late([](void *) { llama_dart_adapter_lora_init(nullptr, ""); }, nullptr);
  late(
      [](void *) {
        llama_dart_mtmd_tokenize(nullptr, nullptr, nullptr, nullptr, 0);
      },
      nullptr);
  late([](void *) { llama_dart_mtmd_encode_chunk(nullptr, nullptr); }, nullptr);
  late(
      [](void *) {
        llama_dart_mtmd_helper_eval_chunks(nullptr, nullptr, nullptr, 0, 0, 1,
                                           true, nullptr);
      },
      nullptr);
  late(
      [](void *) {
        llama_dart_mtmd_helper_eval_chunk_single(nullptr, nullptr, nullptr, 0,
                                                 0, 1, true, nullptr);
      },
      nullptr);
  late(
      [](void *) {
        llama_dart_mtmd_helper_decode_image_chunk(nullptr, nullptr, nullptr,
                                                  nullptr, 0, 0, 1, nullptr,
                                                  nullptr, nullptr);
      },
      nullptr);
  late(
      [](void *) {
        llama_dart_ggml_backend_sched_graph_compute(nullptr, nullptr);
      },
      nullptr);

  while (started.load() < launched) {
    sleep_ms(1);
  }
  sleep_ms(300);
  assert(returned.load() == 0);
  assert(recorded() == std::vector<std::string>({"object"}));
  return 0;
}

#if defined(__APPLE__)
const bool kTeardownRunsAtExit = true;
#else
const bool kTeardownRunsAtExit = false;
#endif

void expect_freed_at_exit() {
  const std::vector<std::string> expected =
      kTeardownRunsAtExit ? kStageOrder : std::vector<std::string>();
  if (recorded() != expected) {
    fprintf(stderr, "exit teardown freed %zu objects, expected %zu\n",
            recorded().size(), expected.size());
    _Exit(EXIT_FAILURE);
  }
}

int test_exit() {
  // Registered first, so it runs after the teardown registered by tracking.
  assert(atexit(expect_freed_at_exit) == 0);
  track_out_of_stage_order();
  return 0;
}

std::string g_system_info;

// Uses the static behind llama_print_system_info, which teardown must find
// alive. Only a sanitizer sees a use of it after its destruction.
void free_and_read_system_info(void *) {
  if (g_system_info != llama_print_system_info()) {
    fprintf(stderr, "the system info changed before teardown\n");
    _Exit(EXIT_FAILURE);
  }
}

// A static that libllamadart creates after the first object was tracked is
// also destroyed only after teardown.
int test_late_static() {
  static char reader[] = "reader";
  track_out_of_stage_order();
  assert(llama_dart_exit_track(reader, free_and_read_system_info,
                               LLAMA_DART_EXIT_STAGE_MODEL));
  // Runs before the teardown registered by tracking, and after the destructor
  // of the static created next.
  assert(atexit(expect_freed_at_exit) == 0);
  g_system_info = llama_print_system_info();
  // The reader needs text that lives on the heap for a sanitizer to see a use
  // after the static is gone. There is none where no backend is loaded.
  assert(!kTeardownRunsAtExit || g_system_info.size() > 64);
  return 0;
}

void add_tensor(gguf_context *gguf, ggml_context *ggml, const std::string &name,
                int64_t columns, int64_t rows, bool is_norm) {
  ggml_tensor *tensor =
      rows == 0 ? ggml_new_tensor_1d(ggml, GGML_TYPE_F32, columns)
                : ggml_new_tensor_2d(ggml, GGML_TYPE_F32, columns, rows);
  ggml_set_name(tensor, name.c_str());
  static uint32_t seed = 1;
  float *data = static_cast<float *>(tensor->data);
  for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
    seed = seed * 1664525u + 1013904223u;
    data[i] = is_norm
                  ? 1.0f
                  : (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 0.1f;
  }
  gguf_add_tensor(gguf, tensor);
}

// The text of a token of the model that make_model writes with a vocabulary:
// two lowercase letters, which a grammar can name without escaping.
std::string vocabulary_piece(llama_token token) {
  return {static_cast<char>('a' + token % 26),
          static_cast<char>('a' + token / 26)};
}

// Writes a two-layer model with random weights: a llama decoder, or a bert
// encoder for llama_encode, which a decoder cannot run. It is large enough to
// load, offload and evaluate, and needs no download. It has no tokenizer
// unless with_vocabulary asks for one, which a grammar sampler needs.
int make_model(const char *path, bool is_encoder,
               bool with_vocabulary = false) {
  const uint32_t vocab_size = 256;
  const uint32_t context_length = 512;
  const uint32_t embedding_length = 128;
  const uint32_t feed_forward_length = 256;
  const uint32_t block_count = 2;
  const std::string arch = is_encoder ? "bert" : "llama";

  gguf_context *gguf = gguf_init_empty();
  const auto set_u32 = [&](const char *key, uint32_t value) {
    gguf_set_val_u32(gguf, (arch + "." + key).c_str(), value);
  };
  gguf_set_val_str(gguf, "general.architecture", arch.c_str());
  gguf_set_val_str(gguf, "general.name", "llamadart exit teardown test");
  set_u32("context_length", context_length);
  set_u32("embedding_length", embedding_length);
  set_u32("block_count", block_count);
  set_u32("feed_forward_length", feed_forward_length);
  set_u32("attention.head_count", 4);
  set_u32("vocab_size", vocab_size);
  gguf_set_val_f32(gguf,
                   (arch + (is_encoder ? ".attention.layer_norm_epsilon"
                                       : ".attention.layer_norm_rms_epsilon"))
                       .c_str(),
                   1e-5f);
  if (with_vocabulary) {
    std::vector<std::string> pieces = {"<unk>", "<s>", "</s>"};
    std::vector<float> scores(vocab_size, 0.0f);
    std::vector<int32_t> types(vocab_size, LLAMA_TOKEN_TYPE_NORMAL);
    types[0] = LLAMA_TOKEN_TYPE_UNKNOWN;
    types[1] = types[2] = LLAMA_TOKEN_TYPE_CONTROL;
    while (pieces.size() < vocab_size) {
      pieces.push_back(
          vocabulary_piece(static_cast<llama_token>(pieces.size())));
    }
    std::vector<const char *> texts;
    for (const std::string &piece : pieces) {
      texts.push_back(piece.c_str());
    }
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", texts.data(),
                     texts.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
                      scores.data(), scores.size());
    gguf_set_arr_data(gguf, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                      types.data(), types.size());
  } else {
    gguf_set_val_str(gguf, "tokenizer.ggml.model", "no_vocab");
  }
  if (is_encoder) {
    gguf_set_val_u32(gguf, "tokenizer.ggml.token_type_count", 1);
  }

  ggml_init_params params{};
  params.mem_size = 16u * 1024 * 1024;
  ggml_context *ggml = ggml_init(params);
  assert(ggml != nullptr);
  const auto add = [&](const std::string &name, int64_t columns, int64_t rows) {
    add_tensor(gguf, ggml, name + ".weight", columns, rows, rows == 0);
  };
  // A norm with a bias, as bert has them.
  const auto add_norm = [&](const std::string &name) {
    add(name, embedding_length, 0);
    add_tensor(gguf, ggml, name + ".bias", embedding_length, 0, false);
  };
  add("token_embd", embedding_length, vocab_size);
  if (is_encoder) {
    add("position_embd", embedding_length, context_length);
    add_norm("token_embd_norm");
  }
  for (uint32_t block = 0; block < block_count; ++block) {
    const std::string prefix = "blk." + std::to_string(block) + ".";
    add(prefix + "attn_q", embedding_length, embedding_length);
    add(prefix + "attn_k", embedding_length, embedding_length);
    add(prefix + "attn_v", embedding_length, embedding_length);
    add(prefix + "attn_output", embedding_length, embedding_length);
    add(prefix + "ffn_up", embedding_length, feed_forward_length);
    add(prefix + "ffn_down", feed_forward_length, embedding_length);
    if (is_encoder) {
      add_norm(prefix + "attn_output_norm");
      add_norm(prefix + "layer_output_norm");
    } else {
      add(prefix + "attn_norm", embedding_length, 0);
      add(prefix + "ffn_norm", embedding_length, 0);
      add(prefix + "ffn_gate", embedding_length, feed_forward_length);
    }
  }
  if (!is_encoder) {
    add("output_norm", embedding_length, 0);
    add("output", embedding_length, vocab_size);
  }

  const bool written = gguf_write_to_file(gguf, path, false);
  gguf_free(gguf);
  ggml_free(ggml);
  return written ? 0 : 1;
}

struct model_options {
  bool tracked = true;
  // Keeps the model and its contexts off every GPU device.
  bool cpu_only = false;
  bool embeddings = false;
  ggml_backend_sched_eval_callback eval_callback = nullptr;
};

struct model_fixture {
  llama_model *model = nullptr;
  llama_context *context = nullptr;
  std::vector<llama_token> tokens;
};

model_fixture load_model(const char *path, model_options options = {}) {
  llama_backend_init();
  model_fixture fixture;
  auto model_params = llama_model_default_params();
  static ggml_backend_dev_t no_devices[] = {nullptr};
  model_params.n_gpu_layers = options.cpu_only ? 0 : 99;
  model_params.devices = options.cpu_only ? no_devices : nullptr;
  model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
  fixture.model = options.tracked
                      ? llama_dart_model_load_from_file(path, model_params)
                      : llama_model_load_from_file(path, model_params);
  assert(fixture.model != nullptr);
  auto context_params = llama_context_default_params();
  context_params.n_ctx = 256;
  context_params.embeddings = options.embeddings;
  context_params.cb_eval = options.eval_callback;
  fixture.context =
      options.tracked
          ? llama_dart_init_from_model(fixture.model, context_params)
          : llama_init_from_model(fixture.model, context_params);
  assert(fixture.context != nullptr);
  // Token ids that every vocabulary has, so no tokenizer is needed.
  for (llama_token token = 1; token <= 32; ++token) {
    fixture.tokens.push_back(token);
  }
  return fixture;
}

llama_batch prompt(model_fixture &fixture) {
  return llama_batch_get_one(fixture.tokens.data(),
                             static_cast<int32_t>(fixture.tokens.size()));
}

void clear_memory(model_fixture &fixture) {
  // llama_memory_clear has no wrapper. Bracketing it keeps the scenarios free
  // of calls that only the settle time covers.
  llama_dart_exit_call_begin();
  llama_memory_clear(llama_get_memory(fixture.context), true);
  llama_dart_exit_call_end();
}

llama_sampler *greedy_sampler() {
  llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  return sampler;
}

// One generation step as the Dart side runs it: decode, then sample.
llama_token decode_and_sample(model_fixture &fixture, llama_sampler *sampler) {
  clear_memory(fixture);
  assert(llama_dart_decode(fixture.context, prompt(fixture)) == 0);
  return llama_dart_sampler_sample(sampler, fixture.context, -1);
}

// Exits with a loaded model and context that nothing frees.
int test_model_idle(const char *path, bool tracked) {
  model_options options;
  options.tracked = tracked;
  model_fixture fixture = load_model(path, options);
  clear_memory(fixture);
  assert(llama_decode(fixture.context, prompt(fixture)) == 0);
  assert(llama_dart_exit_tracked_count() == (tracked ? 2 : 0));
  return 0;
}

int test_model_dispose(const char *path) {
  llama_sampler *sampler = greedy_sampler();
  for (int i = 0; i < 3; ++i) {
    model_fixture fixture = load_model(path);
    assert(decode_and_sample(fixture, sampler) >= 0);
    assert(llama_dart_exit_tracked_count() == 2);
    llama_dart_exit_free(fixture.context);
    llama_dart_exit_free(fixture.model);
    assert(llama_dart_exit_tracked_count() == 0);
  }
  llama_sampler_free(sampler);
  return 0;
}

// A context counts a decode or encode once its backend has finished it, so a
// synchronize that changes the counts shows that work was still pending.
bool has_pending_work(model_fixture &fixture) {
  const llama_perf_context_data before = llama_perf_context(fixture.context);
  llama_synchronize(fixture.context);
  const llama_perf_context_data after = llama_perf_context(fixture.context);
  return before.n_p_eval != after.n_p_eval || before.n_eval != after.n_eval;
}

// The wrapped decode leaves no backend work for a later, unguarded read.
int test_model_sync(const char *path) {
  model_fixture fixture = load_model(path);
  assert(llama_decode(fixture.context, prompt(fixture)) == 0);
  assert(has_pending_work(fixture));

  clear_memory(fixture);
  assert(llama_dart_decode(fixture.context, prompt(fixture)) == 0);
  assert(!has_pending_work(fixture));

  clear_memory(fixture);
  assert(llama_decode(fixture.context, prompt(fixture)) == 0);
  llama_dart_synchronize(fixture.context);
  assert(!has_pending_work(fixture));
  return 0;
}

// Optional real-vocabulary model check for the v0.6.0 process adapter.
int test_model_speculative_process(const char *path) {
  model_fixture fixture = load_model(path);
  assert(llama_dart_decode(fixture.context, prompt(fixture)) == 0);
  // The real draft context must consume a legacy batch, including the
  // implicit position/sequence fields of llama_batch_get_one(). Merely
  // returning true from the process adapter cannot satisfy this assertion.
  llama_dart_speculative_params spec_params{};
  spec_params.type_names = "draft-simple";
  auto *speculative = llama_dart_speculative_init(
      fixture.model, fixture.model, fixture.context,
      llama_context_default_params(), &spec_params);
  assert(speculative != nullptr);
  auto *draft_context = llama_dart_speculative_get_draft_context(speculative);
  assert(draft_context != nullptr);
  assert(llama_dart_speculative_process_batch(speculative, prompt(fixture)));
  assert(llama_memory_seq_pos_max(llama_get_memory(draft_context), 0) ==
         static_cast<llama_pos>(fixture.tokens.size()) - 1);
  auto next_token = fixture.tokens.back();
  assert(llama_dart_speculative_process_batch(
      speculative, llama_batch_get_one(&next_token, 1)));
  assert(llama_memory_seq_pos_max(llama_get_memory(draft_context), 0) ==
         static_cast<llama_pos>(fixture.tokens.size()));
  llama_dart_speculative_free(speculative);
  return 0;
}

// The wrappers pass their arguments and results through.
int test_model_wrappers(const char *path) {
  model_fixture fixture = load_model(path);
  llama_sampler *sampler = greedy_sampler();
  const llama_token sampled = decode_and_sample(fixture, sampler);
  assert(sampled == llama_sampler_sample(sampler, fixture.context, -1));
  llama_sampler_free(sampler);
  const float first_logit = llama_get_logits_ith(fixture.context, -1)[0];

  const size_t size = llama_dart_state_seq_get_size_ext(fixture.context, 0, 0);
  assert(size == llama_state_seq_get_size_ext(fixture.context, 0, 0));
  assert(size > 0);
  std::vector<uint8_t> state(size);
  assert(llama_dart_state_seq_get_data_ext(fixture.context, state.data(),
                                           state.size(), 0, 0) == size);
  clear_memory(fixture);
  assert(llama_dart_state_seq_set_data_ext(fixture.context, state.data(),
                                           state.size(), 0, 0) == size);

  const std::string state_path = std::string(path) + ".state";
  assert(llama_dart_state_save_file(fixture.context, state_path.c_str(),
                                    fixture.tokens.data(),
                                    fixture.tokens.size()));
  clear_memory(fixture);
  std::vector<llama_token> restored(fixture.tokens.size() + 1);
  size_t restored_count = 0;
  assert(llama_dart_state_load_file(fixture.context, state_path.c_str(),
                                    restored.data(), restored.size(),
                                    &restored_count));
  restored.resize(restored_count);
  assert(restored == fixture.tokens);
  assert(remove(state_path.c_str()) == 0);
  assert(llama_get_logits_ith(fixture.context, -1)[0] == first_logit);

  assert(llama_dart_adapter_lora_init(fixture.model, "missing.gguf") ==
         nullptr);
  return 0;
}

int test_graph() {
  llama_backend_init();
  ggml_backend_t backend =
      ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
  assert(backend != nullptr);
  ggml_backend_sched_t sched =
      ggml_backend_sched_new(&backend, nullptr, 1, 16, false, true);
  ggml_init_params params{};
  params.mem_size =
      ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(16, false);
  params.no_alloc = true;
  ggml_context *ggml = ggml_init(params);
  ggml_tensor *left = ggml_new_tensor_1d(ggml, GGML_TYPE_F32, 2);
  ggml_tensor *right = ggml_new_tensor_1d(ggml, GGML_TYPE_F32, 2);
  ggml_tensor *sum = ggml_add(ggml, left, right);
  ggml_cgraph *graph = ggml_new_graph_custom(ggml, 16, false);
  ggml_build_forward_expand(graph, sum);
  assert(ggml_backend_sched_alloc_graph(sched, graph));
  const float left_values[2] = {1.0f, 2.0f};
  const float right_values[2] = {10.0f, 20.0f};
  ggml_backend_tensor_set(left, left_values, 0, sizeof(left_values));
  ggml_backend_tensor_set(right, right_values, 0, sizeof(right_values));

  assert(llama_dart_ggml_backend_sched_graph_compute(sched, graph) ==
         GGML_STATUS_SUCCESS);
  float sum_values[2] = {};
  ggml_backend_tensor_get(sum, sum_values, 0, sizeof(sum_values));
  assert(sum_values[0] == 11.0f && sum_values[1] == 22.0f);

  ggml_backend_sched_free(sched);
  ggml_free(ggml);
  ggml_backend_free(backend);
  return 0;
}

std::mutex g_remaining_mutex;
std::vector<int32_t> g_remaining;

void free_and_count_remaining(void *) {
  const int32_t remaining = llama_dart_exit_tracked_count();
  std::lock_guard<std::mutex> lock(g_remaining_mutex);
  g_remaining.push_back(remaining);
}

// The model, the context and the mtmd context are freed in their stages:
// objects tracked in the stages around them see the right ones still tracked.
int test_model_order(const char *path, const char *mmproj_path) {
  static char scheduler[] = "scheduler";
  static char model_user_early[] = "model-user-early";
  static char model_user_late[] = "model-user-late";
  static char backend[] = "backend";
  const auto track = [](char *object, int32_t stage) {
    assert(llama_dart_exit_track(object, free_and_count_remaining, stage));
  };
  track(scheduler, LLAMA_DART_EXIT_STAGE_SCHEDULER);
  track(model_user_early, LLAMA_DART_EXIT_STAGE_MODEL_USER);
  track(backend, LLAMA_DART_EXIT_STAGE_BACKEND);
  model_fixture fixture = load_model(path);
  int32_t mtmd_count = 0;
  if (mmproj_path != nullptr) {
    const mtmd_context_params params = mtmd_context_params_default();
    assert(llama_dart_mtmd_init_from_file(mmproj_path, fixture.model,
                                          &params) != nullptr);
    mtmd_count = 1;
  }
  track(model_user_late, LLAMA_DART_EXIT_STAGE_MODEL_USER);

  llama_dart_exit_teardown();
  // Within a stage the latest tracked object goes first. The scheduler,
  // tracked before the context, sees everything else; the late model user
  // sees the context gone and the mtmd context still there; the early one
  // sees both gone; the backend, tracked before the model, sees only the
  // model.
  assert(g_remaining ==
         std::vector<int32_t>({5 + mtmd_count, 3 + mtmd_count, 2, 1}));
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

llama_dart_speculative *init_ngram_speculative(model_fixture &fixture) {
  llama_dart_speculative_params params{};
  params.type_names = "ngram-mod";
  params.draft_token_max = -1;
  params.draft_token_min = -1;
  params.draft_min_probability = -1.0f;
  params.draft_split_probability = -1.0f;
  params.ngram_token_min = -1;
  return llama_dart_speculative_init(fixture.model, nullptr, fixture.context,
                                     llama_context_default_params(), &params);
}

std::atomic<int> g_embeddings_seen{-1};

// Freeing speculative state turns the embeddings of its context off again,
// and a decode then leaves no embeddings.
void free_and_decode_embeddings(void *object) {
  model_fixture &fixture = *static_cast<model_fixture *>(object);
  assert(llama_decode(fixture.context, prompt(fixture)) == 0);
  g_embeddings_seen.store(llama_get_embeddings(fixture.context) != nullptr ? 1
                                                                           : 0);
}

// Speculative state is tracked, and teardown frees it before its context.
int test_model_session(const char *path) {
  static model_fixture fixture;
  fixture = load_model(path);
  assert(init_ngram_speculative(fixture) != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);

  llama_set_embeddings(fixture.context, true);
  free_and_decode_embeddings(&fixture);
  assert(g_embeddings_seen.load() == 1);
  clear_memory(fixture);
  // Freed after the speculative state and before the context.
  assert(llama_dart_exit_track(&fixture, free_and_decode_embeddings,
                               LLAMA_DART_EXIT_STAGE_SCHEDULER));

  llama_dart_exit_teardown();
  assert(g_embeddings_seen.load() == 0);
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

struct freed_objects {
  llama_dart_ngram *ngram = nullptr;
  llama_dart_speculative *speculative = nullptr;
  llama_context *context = nullptr;
  mtmd_context *mtmd = nullptr;
  bool ngram_freed = false;
  bool speculative_freed = false;
  bool context_freed = false;
  bool mtmd_freed = false;
};

freed_objects g_freed;

void free_and_check_ngram(void *) {
  g_freed.ngram_freed = is_freed(g_freed.ngram);
}

void free_and_check_speculative(void *) {
  g_freed.speculative_freed = is_freed(g_freed.speculative);
}

void free_and_check_context(void *) {
  g_freed.context_freed = is_freed(g_freed.context);
}

void free_and_check_mtmd(void *) {
  g_freed.mtmd_freed = g_freed.mtmd == nullptr || is_freed(g_freed.mtmd);
}

// Teardown frees the tracked objects themselves. Without Metal nothing else
// shows it: leaving them allocated is then harmless at exit.
int test_model_freed(const char *path, const char *mmproj_path) {
  if (!kSeesFreedBlocks) {
    fprintf(stderr, "skipped: this build cannot tell a freed block\n");
    return 0;
  }
  static char after_ngram[] = "after-ngram";
  static char after_speculative[] = "after-speculative";
  static char after_context[] = "after-context";
  static char after_mtmd[] = "after-mtmd";
  const auto track = [](char *object, void (*check)(void *), int32_t stage) {
    assert(llama_dart_exit_track(object, check, stage));
  };
  // Each check is tracked so that teardown reaches it right after the object
  // it looks at, before anything can allocate the same block again: within a
  // stage the latest tracked object goes first.
  model_fixture fixture = load_model(path);
  g_freed.context = fixture.context;
  g_freed.speculative = init_ngram_speculative(fixture);
  assert(g_freed.speculative != nullptr);
  track(after_ngram, free_and_check_ngram, LLAMA_DART_EXIT_STAGE_SESSION);
  g_freed.ngram = llama_dart_ngram_simple_init(4, 8);
  assert(g_freed.ngram != nullptr);
  track(after_speculative, free_and_check_speculative,
        LLAMA_DART_EXIT_STAGE_SCHEDULER);
  if (mmproj_path != nullptr) {
    const mtmd_context_params params = mtmd_context_params_default();
    g_freed.mtmd =
        llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
    assert(g_freed.mtmd != nullptr && !is_freed(g_freed.mtmd));
  }
  track(after_context, free_and_check_context,
        LLAMA_DART_EXIT_STAGE_MODEL_USER);
  track(after_mtmd, free_and_check_mtmd, LLAMA_DART_EXIT_STAGE_BACKEND);
  assert(!is_freed(g_freed.ngram) && !is_freed(g_freed.speculative));
  assert(!is_freed(fixture.context) && !is_freed(fixture.model));

  llama_dart_exit_teardown();
  assert(is_freed(fixture.model));
  assert(g_freed.ngram_freed);
  assert(g_freed.speculative_freed);
  assert(g_freed.context_freed);
  assert(g_freed.mtmd_freed);
  return 0;
}

// Teardown leaves speculative state alone over a context that an existing
// caller created with the upstream function and may be decoding on. With
// nothing else to free, it does not wait for calls in flight either.
int test_model_session_untracked(const char *path) {
  model_options options;
  options.tracked = false;
  // Metal aborts at exit over a model that is left loaded.
  options.cpu_only = true;
  model_fixture fixture = load_model(path, options);
  assert(init_ngram_speculative(fixture) != nullptr);
  assert(llama_dart_exit_tracked_count() == 1);

  std::thread([] { llama_dart_exit_call_begin(); }).join();
  llama_dart_exit_set_wait_ms(30000);
  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(elapsed_ms(started) < 5000);
  assert(llama_dart_exit_tracked_count() == 1);
  return 0;
}

std::atomic<bool> g_hold_evaluation{false};
std::atomic<bool> g_evaluation_held{false};
std::atomic<bool> g_release_evaluation{false};
std::atomic<bool> g_released_before_teardown{false};
std::atomic<int64_t> g_last_beat_ms{0};

// A thread outside any call in flight blocks in libllamadart once teardown
// has begun, so this beat stops when teardown starts to wait.
void beat_until_teardown() {
  static char untracked[] = "untracked";
  for (;;) {
    g_last_beat_ms.store(now_ms());
    llama_dart_exit_untrack(untracked);
    sleep_ms(1);
  }
}

void start_beat() {
  std::thread(beat_until_teardown).detach();
  while (g_last_beat_ms.load() == 0) {
    sleep_ms(1);
  }
}

// Releases the held evaluation once teardown is waiting for it, and gives up
// after ten seconds.
void release_when_teardown_waits() {
  const int64_t started = now_ms();
  while (now_ms() - g_last_beat_ms.load() < 200) {
    if (now_ms() - started > 10000) {
      g_released_before_teardown.store(true);
      break;
    }
    sleep_ms(1);
  }
  g_release_evaluation.store(true);
}

// Holds the evaluation that calls it until it is released.
bool hold_evaluation(ggml_tensor *, bool, void *) {
  if (g_hold_evaluation.load()) {
    g_evaluation_held.store(true);
    while (!g_release_evaluation.load()) {
      sleep_ms(1);
    }
  }
  return false;
}

void fail_on_pending_work(void *object) {
  if (has_pending_work(*static_cast<model_fixture *>(object))) {
    fprintf(stderr,
            "teardown found no call in flight while the backend had work\n");
    _Exit(EXIT_FAILURE);
  }
}

// The backend wait of a wrapped evaluation is part of its call in flight: an
// evaluation that is still computing when teardown begins has left no work
// pending by the time teardown sees no call in flight and frees.
int test_evaluation_wait(const char *path, void (*evaluate)(model_fixture &),
                         bool embeddings) {
  static model_fixture fixture;
  static void (*held_evaluate)(model_fixture &) = nullptr;
  model_options options;
  options.embeddings = embeddings;
  options.eval_callback = hold_evaluation;
  fixture = load_model(path, options);
  // The first evaluation of a context is counted before it is computed.
  evaluate(fixture);
  assert(!has_pending_work(fixture));
  clear_memory(fixture);

  // The session stage is freed first, as soon as no call is in flight.
  assert(llama_dart_exit_track(&fixture, fail_on_pending_work,
                               LLAMA_DART_EXIT_STAGE_SESSION));
  llama_dart_exit_set_wait_ms(30000);
  held_evaluate = evaluate;
  g_hold_evaluation.store(true);
  std::thread([] { held_evaluate(fixture); }).detach();
  while (!g_evaluation_held.load()) {
    sleep_ms(1);
  }
  start_beat();
  std::thread(release_when_teardown_waits).detach();
  llama_dart_exit_teardown();
  assert(g_release_evaluation.load());
  assert(!g_released_before_teardown.load());
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

void decode_prompt(model_fixture &fixture) {
  assert(llama_dart_decode(fixture.context, prompt(fixture)) == 0);
}

void encode_prompt(model_fixture &fixture) {
  assert(llama_dart_encode(fixture.context, prompt(fixture)) == 0);
}

// Evaluates a text chunk, which needs no mtmd context.
void eval_text_chunk(model_fixture &fixture) {
  mtmd_input_chunks *chunks = mtmd_test_create_input_chunks();
  const mtmd_input_chunk *text = mtmd_input_chunks_get(chunks, 0);
  assert(mtmd_input_chunk_get_type(text) == MTMD_INPUT_CHUNK_TYPE_TEXT);
  llama_pos position = 0;
  assert(llama_dart_mtmd_helper_eval_chunk_single(
             nullptr, fixture.context, text, 0, 0, 16, true, &position) == 0);
  assert(position == 5);
  mtmd_input_chunks_free(chunks);
}

std::atomic<bool> g_exiting{false};

// Runs generation steps until the process is gone. The step that sees the exit
// starts late, when the exit has already reached libllamadart's statics.
void generate_until_exit(model_fixture &fixture) {
  llama_sampler *sampler = greedy_sampler();
  for (;;) {
    llama_dart_exit_call_begin();
    if (g_exiting.load()) {
      sleep_ms(300);
    }
    assert(decode_and_sample(fixture, sampler) >= 0);
    llama_dart_exit_call_end();
  }
}

int exit_while_generating(model_fixture &fixture) {
  static model_fixture *generating = nullptr;
  static std::atomic<int> steps{0};
  generating = &fixture;
  std::thread([] {
    llama_sampler *sampler = greedy_sampler();
    for (int i = 0; i < 3; ++i) {
      assert(decode_and_sample(*generating, sampler) >= 0);
      steps.fetch_add(1);
    }
    llama_sampler_free(sampler);
    steps.fetch_add(1);
    generate_until_exit(*generating);
  }).detach();
  while (steps.load() < 4) {
    sleep_ms(1);
  }
  g_exiting.store(true);
  return 0;
}

// Exits while another thread is generating.
int test_model_decode(const char *path) {
  static model_fixture fixture;
  fixture = load_model(path);
  return exit_while_generating(fixture);
}

// Exits while another thread generates on a model whose load created statics
// in libllamadart that an earlier, different load had not: a CPU-only load,
// then one that maps the file into GPU memory.
int test_model_late_load(const char *path) {
  static model_fixture first;
  static model_fixture second;
  model_options options;
  options.cpu_only = true;
  first = load_model(path, options);
  llama_sampler *sampler = greedy_sampler();
  assert(decode_and_sample(first, sampler) >= 0);
  llama_sampler_free(sampler);
  second = load_model(path);
  return exit_while_generating(second);
}

// Exits while another thread is loading the model.
int test_model_load(const char *path) {
  llama_backend_init();
  static std::atomic<bool> loading{false};
  static const char *model_path = nullptr;
  model_path = path;
  auto params = llama_model_default_params();
  // Holds the load until the process is exiting. Teardown then cancels the
  // load, so the loader does not report progress again.
  params.progress_callback = [](float progress, void *) {
    if (progress <= 0.0f) {
      return true;
    }
    if (loading.exchange(true)) {
      fprintf(stderr, "the load went on after teardown began\n");
      _Exit(EXIT_FAILURE);
    }
    while (!g_exiting.load()) {
      sleep_ms(1);
    }
    sleep_ms(500);
    return true;
  };
  std::thread([params] {
    llama_dart_model_load_from_file(model_path, params);
    fprintf(stderr, "the load returned to its caller after teardown\n");
    _Exit(EXIT_FAILURE);
  }).detach();
  while (!loading.load()) {
    sleep_ms(1);
  }
  g_exiting.store(true);
  return 0;
}

// Teardown waits for a load in flight although nothing is tracked yet.
int test_model_load_wait(const char *path) {
  llama_backend_init();
  static std::atomic<bool> loading{false};
  static std::atomic<bool> held{false};
  static const char *model_path = nullptr;
  model_path = path;
  auto params = llama_model_default_params();
  params.progress_callback = [](float progress, void *) {
    if (progress > 0.0f && !loading.exchange(true)) {
      sleep_ms(400);
      held.store(true);
    }
    return true;
  };
  std::thread([params] {
    llama_dart_model_load_from_file(model_path, params);
  }).detach();
  while (!loading.load()) {
    sleep_ms(1);
  }
  llama_dart_exit_set_wait_ms(30000);
  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  assert(held.load());
  assert(elapsed_ms(started) < 20000);
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

std::atomic<bool> g_free_started{false};
std::atomic<bool> g_free_finished{false};

// A free that the exit catches in flight: it finishes only once teardown has
// begun.
void free_when_teardown_waits(void *) {
  g_free_started.store(true);
  release_when_teardown_waits();
  g_free_finished.store(true);
}

// Teardown waits for a free in flight although nothing is tracked any more.
// The free has untracked its object, and the exit would go on to destroy what
// the free still uses.
int test_free_in_flight() {
  static char object[] = "object";
  assert(llama_dart_exit_track(object, free_when_teardown_waits,
                               LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_exit_set_wait_ms(30000);
  start_beat();
  std::thread([] { llama_dart_exit_free(object); }).detach();
  while (!g_free_started.load()) {
    sleep_ms(1);
  }
  assert(llama_dart_exit_tracked_count() == 0);

  llama_dart_exit_teardown();
  assert(g_free_finished.load());
  assert(!g_released_before_teardown.load());
  return 0;
}

// A free that never finishes holds teardown up for the wait time only.
int test_free_timeout() {
  static char object[] = "object";
  assert(llama_dart_exit_track(
      object,
      [](void *) {
        g_free_started.store(true);
        sleep_ms(600000);
      },
      LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_exit_set_wait_ms(1000);
  std::thread([] { llama_dart_exit_free(object); }).detach();
  while (!g_free_started.load()) {
    sleep_ms(1);
  }

  const auto started = std::chrono::steady_clock::now();
  llama_dart_exit_teardown();
  const int64_t waited = elapsed_ms(started);
  assert(waited >= 900);
  assert(waited < 20000);
  return 0;
}

void free_model_when_teardown_waits(void *object) {
  g_free_started.store(true);
  release_when_teardown_waits();
  llama_model_free(static_cast<llama_model *>(object));
  g_free_finished.store(true);
}

void expect_free_finished_at_exit() {
  if (!g_free_finished.load() || g_released_before_teardown.load()) {
    fprintf(stderr, "the exit did not wait for the free in flight\n");
    _Exit(EXIT_FAILURE);
  }
}

// Exits while another thread frees the model, the only object still tracked.
// Metal aborts in its static destructor when the exit does not wait for the
// free; without Metal the check at exit shows it.
int test_model_free_in_flight(const char *path) {
  // Registered first, so it runs after teardown and after the Metal device,
  // which the load creates, is destroyed.
  assert(atexit(expect_free_finished_at_exit) == 0);
  static model_fixture fixture;
  fixture = load_model(path);
  llama_sampler *sampler = greedy_sampler();
  assert(decode_and_sample(fixture, sampler) >= 0);
  llama_sampler_free(sampler);
  llama_dart_exit_free(fixture.context);
  // Tracked again, with a free that the test can hold.
  assert(llama_dart_exit_track(fixture.model, free_model_when_teardown_waits,
                               LLAMA_DART_EXIT_STAGE_MODEL));
  llama_dart_exit_set_wait_ms(30000);
  start_beat();
  std::thread([] { llama_dart_exit_free(fixture.model); }).detach();
  while (!g_free_started.load()) {
    sleep_ms(1);
  }
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

// Evaluates an image prompt chunk by chunk, as the Dart side does.
int32_t observe_image_batch(llama_batch batch, void *opaque) {
  auto *calls = static_cast<int32_t *>(opaque);
  ++*calls;
  assert(batch.n_tokens > 0 && batch.token == nullptr && batch.embd != nullptr);
  assert(batch.pos != nullptr);
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    assert(batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == 0);
    assert(batch.logits[i] == 0);
  }
  return 0;
}

int32_t reject_image_batch(llama_batch batch, void *opaque) {
  observe_image_batch(batch, opaque);
  return 73;
}

int32_t eval_chunks_one_by_one(mtmd_context *mtmd, model_fixture &fixture,
                               const mtmd_input_chunks *chunks,
                               llama_pos *position) {
  const size_t count = mtmd_input_chunks_size(chunks);
  llama_pos past = 0;
  for (size_t i = 0; i < count; ++i) {
    const mtmd_input_chunk *chunk = mtmd_input_chunks_get(chunks, i);
    int32_t status = 0;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      status = llama_dart_mtmd_helper_eval_chunk_single(
          mtmd, fixture.context, chunk, past, 0, 64, i == count - 1, position);
    } else {
      status = llama_dart_mtmd_encode_chunk(mtmd, chunk);
      if (status == 0) {
        int32_t callback_calls = 0;
        status = llama_dart_mtmd_helper_decode_image_chunk(
            mtmd, fixture.context, chunk, mtmd_get_output_embd(mtmd), past, 0,
            64, position, observe_image_batch, &callback_calls);
        if (status == 0) {
          assert(callback_calls > 0);
        }
      }
    }
    if (status != 0) {
      return status;
    }
    assert(!has_pending_work(fixture));
    past = *position;
  }
  return 0;
}

// Exits with a loaded model, context and mtmd context that nothing frees,
// after an image prompt went through the mtmd wrappers.
int test_model_mtmd(const char *path, const char *mmproj_path) {
  model_fixture fixture = load_model(path);
  const mtmd_context_params params = mtmd_context_params_default();
  assert(llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, nullptr) ==
         nullptr);
  mtmd_context *mtmd =
      llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
  assert(mtmd != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  llama_dart_exit_free(mtmd);
  assert(llama_dart_exit_tracked_count() == 2);
  mtmd = llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
  assert(mtmd != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);

  const std::vector<unsigned char> pixels(224 * 224 * 3, 127);
  mtmd_bitmap *bitmap = mtmd_bitmap_init(224, 224, pixels.data());
  const mtmd_bitmap *bitmaps[1] = {bitmap};
  const std::string text = std::string("Describe ") + mtmd_default_marker();
  const mtmd_input_text input{text.c_str(), text.size(), true, true};
  mtmd_input_chunks *chunks = mtmd_input_chunks_init();
  assert(llama_dart_mtmd_tokenize(mtmd, chunks, &input, bitmaps, 1) == 0);
  assert(mtmd_input_chunks_size(chunks) > 1);

  llama_pos all_at_once = 0;
  clear_memory(fixture);
  assert(llama_dart_mtmd_helper_eval_chunks(mtmd, fixture.context, chunks, 0, 0,
                                            64, true, &all_at_once) == 0);
  assert(all_at_once > 0);
  assert(!has_pending_work(fixture));

  llama_pos one_by_one = 0;
  clear_memory(fixture);
  assert(eval_chunks_one_by_one(mtmd, fixture, chunks, &one_by_one) == 0);
  assert(one_by_one == all_at_once);

  // A failing legacy callback must be observable through the exported wrapper.
  clear_memory(fixture);
  bool rejected_image = false;
  for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
    const auto *chunk = mtmd_input_chunks_get(chunks, i);
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
      assert(llama_dart_mtmd_encode_chunk(mtmd, chunk) == 0);
      int32_t callback_calls = 0;
      llama_pos position = 0;
      assert(llama_dart_mtmd_helper_decode_image_chunk(
          mtmd, fixture.context, chunk, mtmd_get_output_embd(mtmd), 0, 0, 64,
          &position, reject_image_batch, &callback_calls) == 73);
      assert(callback_calls == 1);
      assert(!has_pending_work(fixture));
      rejected_image = true;
      break;
    }
  }
  assert(rejected_image);

  mtmd_input_chunks_free(chunks);
  mtmd_bitmap_free(bitmap);
  return 0;
}

// Exits with MTP state over a tracked context that nothing frees.
int test_model_mtp(const char *path, const char *draft_path) {
  model_fixture fixture = load_model(path);
  llama_model *draft =
      llama_dart_model_load_from_file(draft_path, llama_model_default_params());
  assert(draft != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  llama_dart_mtp *mtp = llama_dart_mtp_init_with_draft_model(
      draft, fixture.context, llama_context_default_params(), 4, 0, 0.0f,
      false);
  assert(mtp != nullptr);
  assert(llama_dart_exit_tracked_count() == 4);
  llama_dart_mtp_free(mtp);
  assert(llama_dart_exit_tracked_count() == 3);
  assert(llama_dart_mtp_init_with_draft_model(draft, fixture.context,
                                              llama_context_default_params(), 4,
                                              0, 0.0f, false) != nullptr);
  return 0;
}

// Exits with TTS state over a tracked context that nothing frees.
int test_model_tts(const char *path, const char *mmproj_path) {
  model_fixture fixture = load_model(path);
  mtmd_context_params params = mtmd_context_params_default();
  params.cb_eval = llama_dart_tts_eval_callback;
  mtmd_context *mtmd =
      llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
  assert(mtmd != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  llama_dart_tts *tts = llama_dart_tts_init(fixture.context, mtmd, &status);
  assert(tts != nullptr && status == LLAMA_DART_TTS_STATUS_OK);
  assert(llama_dart_exit_tracked_count() == 4);
  llama_dart_tts_free(tts);
  assert(llama_dart_exit_tracked_count() == 3);
  assert(llama_dart_tts_init(fixture.context, mtmd, &status) != nullptr);
  return 0;
}

void throw_on_free(void *) { throw std::runtime_error("free failed"); }

// Whether a thread that holds no call in flight sees the registry idle, which
// it is only when every call that threw also ended its call in flight.
// Teardown frees a tracked object only then. It is the last use of the
// registry in a scenario.
bool calls_in_flight_ended() {
  static char probe[] = "probe";
  g_log.clear();
  assert(llama_dart_exit_track(probe, free_named,
                               LLAMA_DART_EXIT_STAGE_SESSION));
  llama_dart_exit_set_wait_ms(100);
  std::thread([] { llama_dart_exit_teardown(); }).join();
  const std::vector<std::string> freed = recorded();
  return !freed.empty() && freed.front() == "probe";
}

// An exception from a tracked object's free function stays inside
// libllamadart and is reported as the calling thread's last error.
int test_barrier_free() {
  static char object[] = "object";
  assert(llama_dart_last_error() == nullptr);
  assert(llama_dart_exit_track(object, throw_on_free,
                               LLAMA_DART_EXIT_STAGE_SESSION));
  llama_dart_exit_free(object);
  assert(llama_dart_last_error() != nullptr);
  assert(std::string(llama_dart_last_error()) == "free failed");
  assert(llama_dart_exit_tracked_count() == 0);

  // A call that catches nothing clears it, and so does the caller.
  assert(llama_dart_exit_track(object, free_named,
                               LLAMA_DART_EXIT_STAGE_SESSION));
  assert(llama_dart_last_error() == nullptr);
  assert(llama_dart_exit_track(object, throw_on_free,
                               LLAMA_DART_EXIT_STAGE_SESSION));
  llama_dart_exit_free(object);
  assert(llama_dart_last_error() != nullptr);
  llama_dart_clear_last_error();
  assert(llama_dart_last_error() == nullptr);

  assert(calls_in_flight_ended());
  return 0;
}

const llama_token kGrammarToken = 3;
const llama_token kRejectedToken = 4;

llama_sampler *grammar_sampler(const llama_vocab *vocab, llama_token token) {
  const std::string grammar =
      "root ::= \"" + vocabulary_piece(token) + "\"";
  llama_sampler *sampler =
      llama_sampler_init_grammar(vocab, grammar.c_str(), "root");
  assert(sampler != nullptr);
  return sampler;
}

bool is_grammar_stack_error(const char *error) {
  return error != nullptr &&
         strstr(error, "Unexpected empty grammar stack") != nullptr;
}

// llama.cpp throws when a grammar sampler accepts a token that its grammar
// rejects. The wrapper reports that as a failure with a message, on the
// thread that made the call only, and the process goes on. Only the
// vocabulary is loaded, which needs no backend.
int test_barrier_grammar(const char *path) {
  llama_backend_init();
  auto model_params = llama_model_default_params();
  model_params.vocab_only = true;
  llama_model *model = llama_dart_model_load_from_file(path, model_params);
  assert(model != nullptr);
  assert(llama_dart_last_error() == nullptr);
  const llama_vocab *vocab = llama_model_get_vocab(model);

  llama_sampler *rejecting = grammar_sampler(vocab, kGrammarToken);
  assert(!llama_dart_sampler_accept(rejecting, kRejectedToken));
  assert(is_grammar_stack_error(llama_dart_last_error()));
  const std::string error = llama_dart_last_error();
  llama_sampler_free(rejecting);

  // Another thread has no error of its own until it catches one, and neither
  // its error nor clearing it changes this thread's.
  std::thread([vocab] {
    assert(llama_dart_last_error() == nullptr);
    llama_sampler *sampler = grammar_sampler(vocab, kRejectedToken);
    assert(!llama_dart_sampler_accept(sampler, kGrammarToken));
    assert(is_grammar_stack_error(llama_dart_last_error()));
    assert(strstr(llama_dart_last_error(),
                  vocabulary_piece(kGrammarToken).c_str()) != nullptr);
    llama_sampler_free(sampler);
    llama_dart_clear_last_error();
    assert(llama_dart_last_error() == nullptr);
  }).join();
  assert(llama_dart_last_error() != nullptr);
  assert(error == llama_dart_last_error());
  assert(error.find(vocabulary_piece(kRejectedToken)) != std::string::npos);

  // A token that the grammar accepts is no error and clears the last one.
  llama_sampler *accepting = grammar_sampler(vocab, kGrammarToken);
  assert(llama_dart_sampler_accept(accepting, kGrammarToken));
  assert(llama_dart_last_error() == nullptr);
  llama_sampler_free(accepting);

  assert(calls_in_flight_ended());
  return 0;
}

std::atomic<int> g_eval_failure{0};

// A graph evaluation callback that throws on request, as a backend does when
// it fails in the middle of a decode.
bool throwing_eval_callback(ggml_tensor *, bool ask, void *) {
  switch (g_eval_failure.exchange(0)) {
  case 1:
    throw std::runtime_error("evaluation failed");
  case 2:
    throw 42;
  default:
    return !ask;
  }
}

// The same through a decode and the sampling that follows it, as generation
// runs them, and for an exception from inside the decode itself.
int test_model_barrier(const char *path) {
  model_options options;
  options.eval_callback = throwing_eval_callback;
  model_fixture fixture = load_model(path, options);
  const llama_vocab *vocab = llama_model_get_vocab(fixture.model);
  llama_sampler *greedy = greedy_sampler();
  const llama_token sampled = decode_and_sample(fixture, greedy);
  assert(sampled >= 0);
  assert(llama_dart_last_error() == nullptr);

  // The greedy sampler selects the token before the grammar sees it, so the
  // chain accepts a token that the grammar rejects.
  const llama_token other =
      sampled == kGrammarToken ? kRejectedToken : kGrammarToken;
  const auto rejecting_chain = [vocab, other] {
    llama_sampler *chain = greedy_sampler();
    llama_sampler_chain_add(chain, grammar_sampler(vocab, other));
    return chain;
  };
  llama_sampler *chain = rejecting_chain();
  assert(llama_dart_sampler_sample(chain, fixture.context, -1) ==
         LLAMA_TOKEN_NULL);
  assert(is_grammar_stack_error(llama_dart_last_error()));
  llama_sampler_free(chain);

  chain = rejecting_chain();
  const int32_t index = -1;
  llama_token draft = 0;
  llama_token accepted = 0;
  assert(llama_dart_sampler_sample_and_accept_n(
             chain, fixture.context, &index, 1, &draft, 0, &accepted, 1) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(is_grammar_stack_error(llama_dart_last_error()));
  llama_sampler_free(chain);

  // A sampler's exception leaves the context usable, and a call that
  // succeeds clears the error.
  assert(llama_dart_sampler_sample(greedy, fixture.context, -1) == sampled);
  assert(llama_dart_last_error() == nullptr);

  // A failure that llama.cpp reports itself is not an exception: it keeps
  // its own return value and leaves no last error.
  llama_token invalid = -2;
  clear_memory(fixture);
  const int32_t status =
      llama_dart_decode(fixture.context, llama_batch_get_one(&invalid, 1));
  assert(status != 0 && status != LLAMA_DART_STATUS_EXCEPTION);
  assert(llama_dart_last_error() == nullptr);
  auto context_params = llama_context_default_params();
  context_params.n_seq_max = 1u << 20;
  assert(llama_dart_init_from_model(fixture.model, context_params) == nullptr);
  assert(llama_dart_last_error() == nullptr);
  assert(llama_dart_exit_tracked_count() == 2);
  assert(decode_and_sample(fixture, greedy) == sampled);
  llama_sampler_free(greedy);

  // An exception from inside a decode. The context is only freed afterwards.
  clear_memory(fixture);
  g_eval_failure.store(1);
  assert(llama_dart_decode(fixture.context, prompt(fixture)) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(std::string(llama_dart_last_error()) == "evaluation failed");
  g_eval_failure.store(2);
  assert(llama_dart_decode(fixture.context, prompt(fixture)) ==
         LLAMA_DART_STATUS_EXCEPTION);
  assert(std::string(llama_dart_last_error()) == "unknown C++ exception");

  assert(calls_in_flight_ended());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::string scenario = argc > 1 ? argv[1] : "";
  const char *first = argc > 2 ? argv[2] : nullptr;
  const char *second = argc > 3 ? argv[3] : nullptr;
  // Keeps llama.cpp's device, load and decode logs out of the test output.
  llama_dart_set_log_level(3);
  if (scenario == "dispose") {
    return test_dispose();
  }
  if (scenario == "order") {
    return test_order();
  }
  if (scenario == "repeat") {
    return test_repeat();
  }
  if (scenario == "in-flight") {
    return test_in_flight();
  }
  if (scenario == "nested") {
    return test_nested();
  }
  if (scenario == "settle") {
    return test_settle();
  }
  if (scenario == "timeout") {
    return test_timeout();
  }
  if (scenario == "idle-wait") {
    return test_idle_wait();
  }
  if (scenario == "own-call") {
    return test_own_call();
  }
  if (scenario == "blocked") {
    return test_blocked();
  }
  if (scenario == "exit") {
    return test_exit();
  }
  if (scenario == "late-static") {
    return test_late_static();
  }
  if (scenario == "free-in-flight") {
    return test_free_in_flight();
  }
  if (scenario == "free-timeout") {
    return test_free_timeout();
  }
  if (scenario == "graph") {
    return test_graph();
  }
  if (scenario == "barrier-free") {
    return test_barrier_free();
  }
  if (first != nullptr) {
    if (scenario == "make-model") {
      return make_model(first, false);
    }
    if (scenario == "make-vocabulary-model") {
      return make_model(first, false, true);
    }
    if (scenario == "barrier-grammar") {
      return test_barrier_grammar(first);
    }
    if (scenario == "model-barrier") {
      return test_model_barrier(first);
    }
    if (scenario == "make-encoder-model") {
      return make_model(first, true);
    }
    if (scenario == "model-decode-wait") {
      return test_evaluation_wait(first, decode_prompt, false);
    }
    if (scenario == "model-encode-wait") {
      return test_evaluation_wait(first, encode_prompt, true);
    }
    if (scenario == "model-mtmd-wait") {
      return test_evaluation_wait(first, eval_text_chunk, false);
    }
    if (scenario == "model-idle") {
      return test_model_idle(first, true);
    }
    if (scenario == "model-idle-untracked") {
      return test_model_idle(first, false);
    }
    if (scenario == "model-dispose") {
      return test_model_dispose(first);
    }
    if (scenario == "model-sync") {
      return test_model_sync(first);
    }
    if (scenario == "model-speculative-process") {
      return test_model_speculative_process(argv[2]);
    }
    if (scenario == "model-wrappers") {
      return test_model_wrappers(first);
    }
    if (scenario == "model-order") {
      return test_model_order(first, second);
    }
    if (scenario == "model-session") {
      return test_model_session(first);
    }
    if (scenario == "model-freed") {
      return test_model_freed(first, second);
    }
    if (scenario == "model-session-untracked") {
      return test_model_session_untracked(first);
    }
    if (scenario == "model-decode") {
      return test_model_decode(first);
    }
    if (scenario == "model-late-load") {
      return test_model_late_load(first);
    }
    if (scenario == "model-load") {
      return test_model_load(first);
    }
    if (scenario == "model-load-wait") {
      return test_model_load_wait(first);
    }
    if (scenario == "model-free-in-flight") {
      return test_model_free_in_flight(first);
    }
    if (second != nullptr) {
      if (scenario == "model-mtmd") {
        return test_model_mtmd(first, second);
      }
      if (scenario == "model-mtp") {
        return test_model_mtp(first, second);
      }
      if (scenario == "model-tts") {
        return test_model_tts(first, second);
      }
    }
  }
  fprintf(stderr, "usage: %s <scenario> [model.gguf [second.gguf]]\n", argv[0]);
  return 2;
}
