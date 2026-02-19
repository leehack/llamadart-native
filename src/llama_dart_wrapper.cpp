#include "llama_dart_wrapper.h"

#include <atomic>
#include <cstring>

#include <string>
#include <vector>

// Global log level (0=none, 1=debug, 2=info, 3=warn, 4=error)
static std::atomic<int> g_dart_log_level{3}; // Default to WARN
// Track last non-CONT severity so continuation lines inherit proper level.
static std::atomic<int> g_last_non_cont_level{GGML_LOG_LEVEL_NONE};

static void llama_dart_native_log_callback(ggml_log_level level,
                                           const char *text, void *user_data) {
  (void)user_data;
  const int configured_level = g_dart_log_level.load(std::memory_order_relaxed);
  // Explicitly suppress all native logs for `none`.
  if (configured_level <= 0) {
    return;
  }

  // ggml levels: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5.
  // CONT lines are continuations of the previous log message; they should
  // follow the previous message severity, not be treated as level 5.
  int effective_level;
  if (level == GGML_LOG_LEVEL_CONT) {
    effective_level = g_last_non_cont_level.load(std::memory_order_relaxed);
  } else {
    effective_level = static_cast<int>(level);
    g_last_non_cont_level.store(effective_level, std::memory_order_relaxed);
  }

  if (effective_level == GGML_LOG_LEVEL_NONE) {
    return;
  }

  if (effective_level >= configured_level) {
    fputs(text, stderr);
    fflush(stderr);
  }
}

extern "C" {

LLAMADART_API void llama_dart_set_log_level(int level) {
  if (level < 0) {
    level = 0;
  } else if (level > 4) {
    level = 4;
  }

  g_dart_log_level.store(level, std::memory_order_relaxed);
  g_last_non_cont_level.store(GGML_LOG_LEVEL_NONE, std::memory_order_relaxed);
  // Set callbacks every time to ensure they are active
  llama_log_set(llama_dart_native_log_callback, nullptr);
  ggml_log_set(llama_dart_native_log_callback, nullptr);
}
}
