#ifdef NDEBUG
#error "Owner tests require active assertions"
#endif
#include "llama_dart_wrapper.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static std::vector<int> freed;
static std::mutex freed_mutex;
struct resource {
  int value;
  void *owner;
  bool reenter;
};
static void free_resource(void *pointer) {
  auto *object = static_cast<resource *>(pointer);
  // A callback may enter the registry and request the same owner again.
  (void)llama_dart_exit_tracked_count();
  {
    std::lock_guard<std::mutex> lock(freed_mutex);
    freed.push_back(object->value);
  }
  if (object->reenter)
    llama_dart_owner_release(object->owner);
  delete object;
}
static resource *track(void *owner, int value, int stage,
                       bool reenter = false) {
  auto *object = new resource{value, owner, reenter};
  assert(llama_dart_owner_track(owner, object, free_resource, stage) == 1);
  return object;
}
static void *active_owner() {
  void *owner = llama_dart_owner_create();
  assert(owner != nullptr);
  assert(llama_dart_owner_activate(owner));
  assert(!llama_dart_owner_activate(owner));
  return owner;
}
static void test_bootstrap_and_stages() {
  assert(llama_dart_owner_count() == 0);
  void *pending = llama_dart_owner_create();
  assert(pending != nullptr);
  llama_dart_owner_abandon(pending);
  assert(!llama_dart_owner_activate(pending));
  llama_dart_owner_release(pending);
  void *first = active_owner();
  assert(first != pending);
  void *second = active_owner();
  llama_dart_owner_abandon(first);
  assert(llama_dart_owner_count() == 2);
  track(first, 5, LLAMA_DART_EXIT_STAGE_MODEL);
  track(first, 2, LLAMA_DART_EXIT_STAGE_CONTEXT);
  track(first, 3, LLAMA_DART_EXIT_STAGE_MODEL_USER);
  track(first, 4, LLAMA_DART_EXIT_STAGE_BACKEND);
  track(first, 1, LLAMA_DART_EXIT_STAGE_SCHEDULER);
  auto *session = track(first, 0, LLAMA_DART_EXIT_STAGE_SESSION, true);
  assert(!llama_dart_exit_untrack(session));
  assert(!llama_dart_exit_track(session, free_resource,
                                LLAMA_DART_EXIT_STAGE_MODEL));
  assert(llama_dart_owner_track(second, session, free_resource, 0) == 0);
  track(second, 6, LLAMA_DART_EXIT_STAGE_SESSION);
  llama_dart_owner_release(first);
  assert((freed == std::vector<int>{0, 1, 2, 3, 4, 5}));
  assert(llama_dart_owner_count() == 1);
  assert(llama_dart_exit_tracked_count() == 1);
  llama_dart_owner_release(first);
  llama_dart_owner_release(second);
  assert(freed.back() == 6 && freed.size() == 7);
  assert(llama_dart_owner_count() == 0);
  assert(llama_dart_exit_tracked_count() == 0);

  // Ordinary free and owner release share one registry identity. Reentrant
  // close from that free must wait until its callback has returned.
  freed.clear();
  first = active_owner();
  auto *ordinary = track(first, 0, 0, true);
  track(first, 1, 5);
  llama_dart_exit_free(ordinary);
  llama_dart_exit_free(ordinary);
  llama_dart_owner_release(first);
  assert((freed == std::vector<int>{0, 1}));
  assert(llama_dart_exit_tracked_count() == 0);
  assert(llama_dart_owner_count() == 0);

  // A raw adoption can be explicitly disowned for its existing raw free.
  first = active_owner();
  auto *raw = track(first, 2, 0);
  assert(llama_dart_owner_untrack(first, raw));
  assert(!llama_dart_owner_untrack(second, raw));
  free_resource(raw);
  llama_dart_owner_release(first);
  assert(freed.back() == 2 && freed.size() == 3);
}

struct gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool entered = false;
  bool proceed = false;
};
static void no_op_free(void *) {}
static bool block_progress(float, void *data) {
  auto &state = *static_cast<gate *>(data);
  std::unique_lock<std::mutex> lock(state.mutex);
  state.entered = true;
  state.changed.notify_all();
  state.changed.wait(lock, [&] { return state.proceed; });
  return true;
}
static void test_model(const char *path) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
  llama_backend_init();
  auto params = llama_model_default_params();
  ggml_backend_dev_t devices[] = {nullptr};
  params.devices = devices;
  params.n_gpu_layers = 0;
  void *pending = llama_dart_owner_create();
  assert(llama_dart_owner_model_load_from_file(pending, path, params) ==
         nullptr);
  assert(std::strstr(llama_dart_last_error(), "not active") != nullptr);
  assert(llama_dart_exit_tracked_count() == 0);
  llama_dart_owner_abandon(pending);
  void *owner = active_owner();
  assert(llama_dart_owner_model_load_from_file(
             owner, "/nonexistent/owner-test.gguf", params) == nullptr);
  assert(llama_dart_exit_tracked_count() == 0);
  auto *model = llama_dart_owner_model_load_from_file(owner, path, params);
  assert(model != nullptr);
  auto context_params = llama_context_default_params();
  context_params.n_ctx = 256;
  context_params.n_batch = context_params.n_ubatch = 32;
  context_params.n_threads = context_params.n_threads_batch = 2;
  void *foreign = active_owner();
  assert(llama_dart_owner_init_from_model(foreign, model, context_params) ==
         nullptr);
  auto *context =
      llama_dart_owner_init_from_model(owner, model, context_params);
  assert(context != nullptr);
  llama_dart_speculative_params spec_params{};
  spec_params.type_names = "ngram-simple";
  auto *foreign_model =
      llama_dart_owner_model_load_from_file(foreign, path, params);
  assert(foreign_model != nullptr);
  assert(llama_dart_speculative_init(foreign_model, nullptr, context,
                                     context_params, &spec_params) == nullptr);
  assert(llama_dart_speculative_init(model, foreign_model, context,
                                     context_params, &spec_params) == nullptr);
  llama_dart_exit_free(foreign_model);
  auto *session = llama_dart_speculative_init(model, nullptr, context,
                                              context_params, &spec_params);
  assert(session != nullptr);
  assert(llama_dart_exit_tracked_count() == 3);
  llama_dart_owner_release(owner);
  assert(llama_dart_exit_tracked_count() == 0);
  llama_dart_owner_release(foreign);

  // Close while the actual native model creator is admitted. The close cannot
  // finish before the creator publishes-or-frees, even with no prior payload.
  gate state;
  params.progress_callback = block_progress;
  params.progress_callback_user_data = &state;
  owner = active_owner();
  std::atomic<llama_model *> result{nullptr};
  std::thread creator([&] {
    result = llama_dart_owner_model_load_from_file(owner, path, params);
  });
  {
    std::unique_lock<std::mutex> lock(state.mutex);
    assert(state.changed.wait_for(lock, std::chrono::seconds(10),
                                  [&] { return state.entered; }));
  }
  std::atomic<bool> close_started{false}, close_finished{false};
  std::thread closer([&] {
    close_started = true;
    llama_dart_owner_release(owner);
    close_finished = true;
  });
  while (!close_started)
    std::this_thread::yield();
  // Rejected admission proves the closer actually reached its transition;
  // merely observing a thread-start flag would leave a scheduling race.
  while (llama_dart_owner_track(owner, &state, no_op_free, 0) != 0) {
    llama_dart_owner_untrack(owner, &state);
    std::this_thread::yield();
  }
  assert(llama_dart_owner_count() == 1);
  assert(!close_finished);
  {
    std::lock_guard<std::mutex> lock(state.mutex);
    state.proceed = true;
    state.changed.notify_all();
  }
  creator.join();
  closer.join();
  assert(result == nullptr);
  assert(llama_dart_exit_tracked_count() == 0);
  assert(llama_dart_owner_count() == 0);
  llama_backend_free();
}
static void slow_free(void *pointer) {
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  free_resource(pointer);
}
static void test_free_during_owner_drain() {
  freed.clear();
  void *owner = active_owner();
  auto *context = new resource{2, owner, false};
  assert(llama_dart_owner_track(owner, context, slow_free, 2) == 1);
  auto *model = track(owner, 5, 5);
  std::thread closer([=] { llama_dart_owner_release(owner); });
  while (llama_dart_exit_tracked_count() == 2)
    std::this_thread::yield();
  std::thread ordinary([=] { llama_dart_exit_free(model); });
  closer.join();
  ordinary.join();
  assert((freed == std::vector<int>{2, 5}));
  assert(llama_dart_exit_tracked_count() == 0);
}

static void test_process_exit(bool race) {
  void *owner = active_owner();
  auto *first = new resource{0, owner, true};
  assert(llama_dart_owner_track(owner, first, race ? slow_free : free_resource,
                                0) == 1);
  track(owner, 1, LLAMA_DART_EXIT_STAGE_MODEL);
  if (race) {
    std::thread closer([=] { llama_dart_owner_release(owner); });
    // Wait for the first callback to be claimed, then process teardown must
    // wait for the owner drain rather than reuse either cleanup identity.
    while (llama_dart_exit_tracked_count() == 2)
      std::this_thread::yield();
    llama_dart_exit_teardown();
    assert((freed == std::vector<int>{0, 1}));
    std::_Exit(0);
  }
  llama_dart_exit_teardown();
  assert((freed == std::vector<int>{0, 1}));
  std::_Exit(0);
}

int main(int argc, char **argv) {
  if (argc == 2 && std::strcmp(argv[1], "process-exit") == 0)
    test_process_exit(false);
  if (argc == 2 && std::strcmp(argv[1], "process-exit-race") == 0)
    test_process_exit(true);
  test_bootstrap_and_stages();
  test_free_during_owner_drain();
  if (argc == 2)
    test_model(argv[1]);
  else
    assert(argc == 1);
  return 0;
}
