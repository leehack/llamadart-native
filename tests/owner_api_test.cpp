#ifdef NDEBUG
#error "Owner tests require active assertions"
#endif
#include "llama_dart_speculative_internal.h"
#include "llama_dart_tts_internal.h"
#include "llama_dart_wrapper.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
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
static void test_dependency_ownership(const char *scenario) {
  void *owner = active_owner(), *foreign = active_owner();
  int context = 1, model = 2;
  const bool all_unowned = std::strcmp(scenario, "dependency-all-unowned") == 0;
  const bool unowned_first = std::strcmp(scenario, "dependency-unowned-first") == 0;
  const bool owned_first = std::strcmp(scenario, "dependency-owned-first") == 0;
  const bool same_owner = std::strcmp(scenario, "dependency-same-owner") == 0;
  const bool foreign_owner = std::strcmp(scenario, "dependency-foreign-owner") == 0;
  assert(all_unowned || unowned_first || owned_first || same_owner || foreign_owner);
  if (all_unowned || unowned_first) {
    assert(llama_dart_exit_track(&context, no_op_free, LLAMA_DART_EXIT_STAGE_CONTEXT));
  } else {
    assert(llama_dart_owner_track(owner, &context, no_op_free, LLAMA_DART_EXIT_STAGE_CONTEXT));
  }
  if (all_unowned || owned_first) {
    assert(llama_dart_exit_track(&model, no_op_free, LLAMA_DART_EXIT_STAGE_MODEL));
  } else {
    assert(llama_dart_owner_track(foreign_owner ? foreign : owner, &model,
                                  no_op_free, LLAMA_DART_EXIT_STAGE_MODEL));
  }
  auto params = llama_context_default_params();
  llama_dart_speculative_params spec{};
  // Resolve the ownership boundary without dereferencing these sentinel inputs.
  spec.type_names = "none";
  llama_dart_clear_last_error();
  assert(!llama_dart_speculative_init(reinterpret_cast<llama_model *>(&model),
      nullptr, reinterpret_cast<llama_context *>(&context), params, &spec));
  const char *error = llama_dart_last_error();
  if (unowned_first || owned_first || foreign_owner) {
    assert(error && std::strstr(error, "dependency belongs to another owner"));
  } else {
    assert(error == nullptr);
  }
  llama_dart_exit_free(&context);
  llama_dart_exit_free(&model);
  llama_dart_owner_release(owner);
  llama_dart_owner_release(foreign);
  assert(llama_dart_owner_count() == 0 && llama_dart_exit_tracked_count() == 0);
}
static bool block_progress(float, void *data) {
  auto &state = *static_cast<gate *>(data);
  std::unique_lock<std::mutex> lock(state.mutex);
  state.entered = true;
  state.changed.notify_all();
  state.changed.wait(lock, [&] { return state.proceed; });
  return true;
}
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
// MTP and n-gram now share the generic lifetime. The model-free callback
// races above cover that allocation family; this case verifies real MTP
// construction inherits the owner and drains its actual draft context.
static void test_mtp_model(const char *path, const char *draft_path) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
  llama_backend_init();
  void *owner = active_owner();
  auto model_params = llama_model_default_params();
  ggml_backend_dev_t devices[] = {nullptr};
  model_params.devices = devices;
  model_params.n_gpu_layers = 0;
  auto *model = llama_dart_owner_model_load_from_file(owner, path, model_params);
  model_params.load_mtp = true;
  auto *draft_model = llama_dart_owner_model_load_from_file(
      owner, draft_path, model_params);
  assert(model != nullptr && draft_model != nullptr);
  auto params = llama_context_default_params();
  params.n_ctx = 256;
  params.n_batch = params.n_ubatch = 32;
  params.n_threads = params.n_threads_batch = 2;
  auto *context = llama_dart_owner_init_from_model(owner, model, params);
  assert(context != nullptr);
  llama_dart_speculative_params spec_params{};
  spec_params.type_names = "draft-mtp";
  spec_params.draft_token_max = 4;
  auto *session = llama_dart_speculative_init(
      model, draft_model, context, params, &spec_params);
  assert(session != nullptr);
  assert(llama_dart_speculative_get_draft_context(session) != nullptr);
  assert(llama_dart_speculative_need_embd_nextn(session));
  assert(llama_dart_exit_tracked_count() == 4);
  llama_dart_speculative_free(session);
  assert(llama_dart_exit_tracked_count() == 3);
  session = llama_dart_speculative_init(
      model, draft_model, context, params, &spec_params);
  assert(session != nullptr && llama_dart_exit_tracked_count() == 4);
  std::thread closer([&] { llama_dart_owner_release(owner); });
  while (llama_dart_exit_tracked_count() == 4)
    std::this_thread::yield();
  llama_dart_speculative_free(session);
  closer.join();
  assert(llama_dart_owner_count() == 0);
  assert(llama_dart_exit_tracked_count() == 0);
  assert(llama_dart_internal::live_session_count() == 0);
  llama_backend_free();
}

#endif

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
  std::thread owner_closer([&] { llama_dart_owner_release(owner); });
  while (llama_dart_exit_tracked_count() == 3)
    std::this_thread::yield();
  llama_dart_speculative_free(session);
  owner_closer.join();
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

// A registered callback may delegate to its public specialized free. A late
// caller on another thread must not steal that callback's destruction
// authority.
template <typename Session, void (*Free)(Session *)> struct session_case {
  inline static gate *callback_gate = nullptr;
  inline static int callback_count = 0;
  static void delegated_free(void *pointer) {
    ++callback_count;
    if (callback_gate != nullptr) {
      auto &state = *callback_gate;
      std::unique_lock<std::mutex> lock(state.mutex);
      state.entered = true;
      state.changed.notify_all();
      state.changed.wait(lock, [&] { return state.proceed; });
    }
    Free(static_cast<Session *>(pointer));
    Free(static_cast<Session *>(pointer));
  }
  static void destructor_only(void *pointer) {
    static_cast<Session *>(pointer)->~Session();
  }
  static void run(bool owner_drain) {
    const auto baseline = llama_dart_internal::live_session_count();
    auto *legacy = new Session();
    Free(legacy);
    Free(legacy);
    assert(llama_dart_internal::live_session_count() == baseline);

    legacy = new Session();
    assert(llama_dart_exit_track(legacy, delegated_free, 0));
    assert(llama_dart_exit_untrack(legacy));
    Free(legacy);
    assert(llama_dart_internal::live_session_count() == baseline);

    void *owner = active_owner();
    auto *session = new Session();
    assert(llama_dart_owner_track(owner, session, delegated_free, 0) == 1);
    gate state;
    callback_gate = &state;
    callback_count = 0;
    std::thread closer([&] {
      if (owner_drain)
        llama_dart_owner_release(owner);
      else
        llama_dart_exit_free(session);
    });
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      assert(state.changed.wait_for(lock, std::chrono::seconds(10),
                                    [&] { return state.entered; }));
    }
    assert(llama_dart_exit_tracked_count() == 0);
    Free(session);
    assert(llama_dart_internal::live_session_count() == baseline + 1);
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.proceed = true;
      state.changed.notify_all();
    }
    closer.join();
    callback_gate = nullptr;
    assert(callback_count == 1);
    assert(llama_dart_internal::live_session_count() == baseline);
    Free(session);
    llama_dart_owner_release(owner);

    // A fresh valid allocation at the identical address must not inherit a
    // retired pointer's claim. Retain the storage to make reuse deterministic.
    void *storage = ::operator new(sizeof(Session));
    auto *first = new (storage) Session();
    owner = active_owner();
    assert(llama_dart_owner_track(owner, first, destructor_only, 0) == 1);
    llama_dart_owner_release(owner);
    auto *second = new (storage) Session();
    Free(second);
    assert(llama_dart_internal::live_session_count() == baseline);
  }
};
using tts_case = session_case<llama_dart_tts, llama_dart_tts_free>;
using speculative_case =
    session_case<llama_dart_speculative, llama_dart_speculative_free>;
static void test_specialized_frees() {
  for (bool owner_drain : {false, true}) {
    tts_case::run(owner_drain);
    speculative_case::run(owner_drain);
  }
}
static void test_specialized_process_exit() {
  void *owner = active_owner();
  assert(llama_dart_owner_track(owner, new llama_dart_tts(),
                                tts_case::delegated_free, 0) == 1);
  assert(llama_dart_owner_track(owner, new llama_dart_speculative(),
                                speculative_case::delegated_free, 0) == 1);
  llama_dart_exit_teardown();
  assert(llama_dart_internal::live_session_count() == 0);
  assert(tts_case::callback_count == 1 &&
         speculative_case::callback_count == 1);
  std::_Exit(0);
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
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
  if (argc == 4 && std::strcmp(argv[1], "mtp-model") == 0) {
    test_mtp_model(argv[2], argv[3]);
    return 0;
  }
#endif
  if (argc == 2 && std::strncmp(argv[1], "dependency-", 11) == 0) {
    test_dependency_ownership(argv[1]);
    return 0;
  }
  if (argc == 2 && std::strcmp(argv[1], "specialized-process-exit") == 0)
    test_specialized_process_exit();
  if (argc == 2 && std::strcmp(argv[1], "process-exit") == 0)
    test_process_exit(false);
  if (argc == 2 && std::strcmp(argv[1], "process-exit-race") == 0)
    test_process_exit(true);
  test_specialized_frees();
  test_bootstrap_and_stages();
  test_free_during_owner_drain();
  if (argc == 2)
    test_model(argv[1]);
  else
    assert(argc == 1);
  return 0;
}
