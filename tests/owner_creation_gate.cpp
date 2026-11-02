// A test-only FFI gate after the real owner-aware creator has returned. A
// pending Dart kill then lands before any Dart-side hold or result handling.
#include "llama_dart_wrapper.h"
#include <atomic>
#include <chrono>
#include <thread>

#if defined(_WIN32)
#define GATE_API __declspec(dllexport)
#else
#define GATE_API __attribute__((visibility("default")))
#endif
static std::atomic<bool> waiting{false};
static std::atomic<bool> resumed{false};
static void gate() {
  waiting = true;
  while (!resumed)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
extern "C" {
GATE_API void owner_gate_reset() {
  waiting = false;
  resumed = false;
}
GATE_API bool owner_gate_waiting() { return waiting; }
GATE_API void owner_gate_resume() { resumed = true; }
GATE_API llama_model *owner_gate_model(
    llama_model *(*create)(void *, const char *, llama_model_params),
    void *owner, const char *path, llama_model_params params) {
  auto *result = create(owner, path, params);
  if (result != nullptr)
    gate();
  return result;
}
GATE_API llama_context *owner_gate_context(
    llama_context *(*create)(void *, llama_model *, llama_context_params),
    void *owner, llama_model *model, llama_context_params params) {
  auto *result = create(owner, model, params);
  if (result != nullptr)
    gate();
  return result;
}
}
