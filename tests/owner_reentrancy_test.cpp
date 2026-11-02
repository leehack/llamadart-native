#ifdef NDEBUG
#error "Owner tests require active assertions"
#endif
#include "llama_dart_wrapper.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

static void *active_owner() {
  auto *owner = llama_dart_owner_create();
  assert(owner && llama_dart_owner_activate(owner));
  return owner;
}
static std::atomic<int> calls{0};
struct peer { void *owner; void *next; };
static void close_peer(void *p) {
  auto *value = static_cast<peer *>(p);
  ++calls;
  llama_dart_owner_release(value->next);
  delete value;
}
static void nested(bool ordinary) {
  void *owners[] = {active_owner(), active_owner(), active_owner()};
  auto *first = new peer{owners[0], owners[1]};
  assert(llama_dart_owner_track(owners[0], first, close_peer, 0));
  for (int i = 1; i < 3; ++i) {
    assert(llama_dart_owner_track(owners[i],
        new peer{owners[i], owners[(i + 1) % 3]}, close_peer, 0));
  }
  if (ordinary) llama_dart_exit_free(first);
  else llama_dart_owner_release(owners[0]);
  assert(calls == 3 && llama_dart_owner_count() == 0);
  assert(llama_dart_exit_tracked_count() == 0);
}

static std::atomic<int> entered{0};
static void concurrent_peer(void *p) {
  ++entered;
  while (entered.load() != 2) std::this_thread::yield();
  close_peer(p);
}
static void concurrent(bool ordinary) {
  void *a = active_owner(), *b = active_owner();
  auto *first = new peer{a, b}, *second = new peer{b, a};
  assert(llama_dart_owner_track(a, first, concurrent_peer, 0));
  assert(llama_dart_owner_track(b, second, concurrent_peer, 0));
  std::thread one([&] {
    if (ordinary) llama_dart_exit_free(first);
    else llama_dart_owner_release(a);
  });
  std::thread two([&] {
    if (ordinary) llama_dart_exit_free(second);
    else llama_dart_owner_release(b);
  });
  one.join(); two.join();
  assert(calls == 2 && llama_dart_owner_count() == 0);
  assert(llama_dart_exit_tracked_count() == 0);
}

static std::vector<int> order;
static void *ancestor_model;
static void *ancestor_owner;
static void *nested_owner;
static void release_ancestor_model(void *) {
  order.push_back(1);
  llama_dart_owner_release(ancestor_owner);
  llama_dart_exit_free(ancestor_model);
  order.push_back(2);
}
static void model_free(void *p) {
  order.push_back(4); delete static_cast<int *>(p);
}
static void context_free(void *p) {
  order.push_back(0);
  llama_dart_owner_release(nested_owner);
  order.push_back(3); delete static_cast<int *>(p);
}
static void ancestor_free(bool ordinary) {
  ancestor_owner = active_owner(); nested_owner = active_owner();
  ancestor_model = new int(1);
  assert(llama_dart_owner_track(ancestor_owner, ancestor_model, model_free, 5));
  auto *context = new int(2);
  assert(llama_dart_owner_track(ancestor_owner, context, context_free, 2));
  assert(llama_dart_owner_track(nested_owner, nested_owner, release_ancestor_model, 0));
  if (ordinary) llama_dart_exit_free(context);
  else llama_dart_owner_release(ancestor_owner);
  assert((order == std::vector<int>{0, 1, 2, 3, 4}));
  assert(llama_dart_owner_count() == 0);
}

struct creation_state {
  const char *path;
  void *a;
  void *b;
  bool called = false;
  std::mutex mutex;
  std::condition_variable changed;
  bool ready = false;
  bool proceed = false;
};
static bool close_creator(float, void *data) {
  auto &state = *static_cast<creation_state *>(data);
  llama_dart_owner_release(state.a);
  return true;
}
static bool enter_creator(float, void *data) {
  auto &state = *static_cast<creation_state *>(data);
  if (state.called) return true;
  state.called = true;
  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;
  ggml_backend_dev_t devices[] = {nullptr}; params.devices = devices;
  params.progress_callback = close_creator;
  params.progress_callback_user_data = &state;
  auto *model = llama_dart_owner_model_load_from_file(state.b, state.path, params);
  assert(model);
  return true;
}
static bool wait_creator(float, void *data) {
  auto &state = *static_cast<creation_state *>(data);
  std::unique_lock<std::mutex> lock(state.mutex);
  state.ready = true; state.changed.notify_all();
  assert(state.changed.wait_for(lock, std::chrono::seconds(10), [&] { return state.proceed; }));
  return true;
}
static void creator(const char *path, bool nested_creation) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND));
#endif
  llama_backend_init();
  creation_state state{path, active_owner(), active_owner()};
  auto params = llama_model_default_params();
  params.n_gpu_layers = 0;
  ggml_backend_dev_t devices[] = {nullptr}; params.devices = devices;
  params.progress_callback_user_data = &state;
  if (nested_creation) {
    params.progress_callback = enter_creator;
    assert(!llama_dart_owner_model_load_from_file(state.a, path, params));
    assert(state.called);
    assert(llama_dart_owner_count() == 1);
    llama_dart_owner_release(state.b);
  } else {
    params.progress_callback = wait_creator;
    std::thread thread([&] {
      assert(!llama_dart_owner_model_load_from_file(state.b, path, params));
    });
    {
      std::unique_lock<std::mutex> lock(state.mutex);
      assert(state.changed.wait_for(lock, std::chrono::seconds(10), [&] { return state.ready; }));
    }
    assert(llama_dart_owner_track(state.a, new peer{state.a, state.b}, close_peer, 0));
    llama_dart_owner_release(state.a);
    // B has only an admitted creator, no active drainer. Its scope must finish
    // the deferred request without requiring a second explicit release.
    assert(llama_dart_owner_count() == 1);
    {
      std::lock_guard<std::mutex> lock(state.mutex);
      state.proceed = true; state.changed.notify_all();
    }
    thread.join();
  }
  assert(llama_dart_owner_count() == 0 && llama_dart_exit_tracked_count() == 0);
  llama_backend_free();
}
int main(int argc, char **argv) {
  assert(argc >= 2);
  if (!std::strcmp(argv[1], "nested")) nested(false);
  else if (!std::strcmp(argv[1], "ordinary")) nested(true);
  else if (!std::strcmp(argv[1], "concurrent-drain")) concurrent(false);
  else if (!std::strcmp(argv[1], "concurrent-free")) concurrent(true);
  else if (!std::strcmp(argv[1], "ancestor-free")) ancestor_free(false);
  else if (!std::strcmp(argv[1], "ancestor-ordinary-free")) ancestor_free(true);
  else if (!std::strcmp(argv[1], "creator-ancestor")) { assert(argc == 3); creator(argv[2], true); }
  else if (!std::strcmp(argv[1], "creator-pending")) { assert(argc == 3); creator(argv[2], false); }
  else assert(false);
}
