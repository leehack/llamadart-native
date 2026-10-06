#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_wrapper.h"

#include "mtmd.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

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
  const int64_t waited = elapsed_ms(started);
  assert(waited >= 300);
  assert(waited < 20000);

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
  llama_dart_exit_set_wait_ms(200);

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
  assert(waited >= 150);
  assert(waited < 20000);
  assert(recorded().empty());
  assert(llama_dart_exit_tracked_count() ==
         static_cast<int32_t>(kStageOrder.size()));
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
  const auto late = [](void (*call)(void *), void *argument) {
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

  while (started.load() < 10) {
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

struct model_fixture {
  llama_model *model = nullptr;
  llama_context *context = nullptr;
  std::vector<llama_token> tokens;
};

model_fixture load_model(const char *path, bool tracked) {
  llama_backend_init();
  model_fixture fixture;
  const auto model_params = llama_model_default_params();
  fixture.model = tracked ? llama_dart_model_load_from_file(path, model_params)
                          : llama_model_load_from_file(path, model_params);
  assert(fixture.model != nullptr);
  auto context_params = llama_context_default_params();
  context_params.n_ctx = 512;
  fixture.context =
      tracked ? llama_dart_init_from_model(fixture.model, context_params)
              : llama_init_from_model(fixture.model, context_params);
  assert(fixture.context != nullptr);

  const llama_vocab *vocab = llama_model_get_vocab(fixture.model);
  const char *text = "The quick brown fox jumps over the lazy dog.";
  fixture.tokens.resize(64);
  const int32_t count = llama_tokenize(
      vocab, text, static_cast<int32_t>(strlen(text)), fixture.tokens.data(),
      static_cast<int32_t>(fixture.tokens.size()), true, true);
  assert(count > 0);
  fixture.tokens.resize(static_cast<size_t>(count));
  return fixture;
}

int32_t decode_prompt(model_fixture &fixture, bool tracked) {
  // The calls llama_dart_decode does not cover are in flight as well.
  llama_dart_exit_call_begin();
  llama_memory_clear(llama_get_memory(fixture.context), true);
  const llama_batch batch = llama_batch_get_one(
      fixture.tokens.data(), static_cast<int32_t>(fixture.tokens.size()));
  const int32_t status = tracked ? llama_dart_decode(fixture.context, batch)
                                 : llama_decode(fixture.context, batch);
  llama_dart_exit_call_end();
  return status;
}

// Exits with a loaded model and context that nothing frees.
int test_model_idle(const char *path, bool tracked) {
  model_fixture fixture = load_model(path, tracked);
  assert(decode_prompt(fixture, tracked) == 0);
  assert(llama_dart_exit_tracked_count() == (tracked ? 2 : 0));
  return 0;
}

int test_model_dispose(const char *path) {
  for (int i = 0; i < 3; ++i) {
    model_fixture fixture = load_model(path, true);
    assert(decode_prompt(fixture, true) == 0);
    assert(llama_dart_exit_tracked_count() == 2);
    llama_dart_exit_free(fixture.context);
    llama_dart_exit_free(fixture.model);
    assert(llama_dart_exit_tracked_count() == 0);
  }
  return 0;
}

// Exits with a loaded model, context and mtmd context that nothing frees.
int test_model_mtmd(const char *path, const char *mmproj_path) {
  model_fixture fixture = load_model(path, true);
  const mtmd_context_params params = mtmd_context_params_default();
  assert(llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, nullptr) ==
         nullptr);
  mtmd_context *mtmd =
      llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params);
  assert(mtmd != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  llama_dart_exit_free(mtmd);
  assert(llama_dart_exit_tracked_count() == 2);
  assert(llama_dart_mtmd_init_from_file(mmproj_path, fixture.model, &params) !=
         nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  return 0;
}

// Exits while another thread is decoding.
int test_model_decode(const char *path) {
  static model_fixture fixture;
  fixture = load_model(path, true);
  static std::atomic<int> decoded{0};
  std::thread([] {
    for (;;) {
      assert(decode_prompt(fixture, true) == 0);
      decoded.fetch_add(1);
    }
  }).detach();
  while (decoded.load() < 3) {
    sleep_ms(1);
  }
  return 0;
}

// Exits while another thread is loading the model.
int test_model_load(const char *path) {
  llama_backend_init();
  static std::atomic<bool> loading{false};
  static std::atomic<bool> exiting{false};
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
    while (!exiting.load()) {
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
  exiting.store(true);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  const std::string scenario = argc > 1 ? argv[1] : "";
  const char *model = argc > 2 ? argv[2] : nullptr;
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
  if (scenario == "blocked") {
    return test_blocked();
  }
  if (scenario == "exit") {
    return test_exit();
  }
  if (model != nullptr) {
    if (scenario == "model-idle") {
      return test_model_idle(model, true);
    }
    if (scenario == "model-idle-untracked") {
      return test_model_idle(model, false);
    }
    if (scenario == "model-dispose") {
      return test_model_dispose(model);
    }
    if (scenario == "model-decode") {
      return test_model_decode(model);
    }
    if (scenario == "model-load") {
      return test_model_load(model);
    }
    if (scenario == "model-mtmd" && argc > 3) {
      return test_model_mtmd(model, argv[3]);
    }
  }
  fprintf(stderr, "usage: %s <scenario> [model.gguf [mmproj.gguf]]\n", argv[0]);
  return 2;
}
