#pragma once

#include "llama_dart_tts_eval_internal.h"

#include <atomic>
#include <cstdint>

static inline bool llama_dart_cancel_flag_set(const int8_t *flag) {
  using atomic_flag_byte = std::atomic<int8_t>;
  static_assert(sizeof(atomic_flag_byte) == sizeof(int8_t) &&
                    alignof(atomic_flag_byte) == alignof(int8_t) &&
                    atomic_flag_byte::is_always_lock_free,
                "a caller-owned cancel byte must be readable atomically");
  return flag != nullptr &&
         reinterpret_cast<const atomic_flag_byte *>(flag)->load(
             std::memory_order_relaxed) != 0;
}

struct llama_dart_cancel_eval_state {
  explicit llama_dart_cancel_eval_state(const int8_t *cancel_flag)
      : flag(cancel_flag) {}

  const int8_t *flag;
  std::atomic<bool> observed{false};
  double budget = llama_dart_tts_eval_budget;
  double work = 0.0;
};

// A chunk boundary splits any backend fusion that spans it, and Metal, CUDA
// and Vulkan fuse a MUL_MAT with the bias ADD after it. Boundaries therefore
// fall only after attention, which no backend fuses with the nodes after it.
static inline bool llama_dart_cancel_eval_boundary_op(const ggml_tensor *node) {
  return node->op == GGML_OP_FLASH_ATTN_EXT || node->op == GGML_OP_SOFT_MAX;
}

// Once cancelled, a split's compute ends after the next computed node,
// wherever that falls: the output of a cancelled encode is discarded.
static inline bool
llama_dart_cancel_eval_answer(llama_dart_cancel_eval_state *state,
                              const ggml_tensor *node, bool ask) {
  const bool cancelled =
      llama_dart_tts_cancel_observed(&state->observed, state->flag);
  if (!ask) {
    return !cancelled;
  }
  if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
    return false;
  }
  if (cancelled) {
    return true;
  }
  state->work += llama_dart_tts_eval_node_work(node);
  if (state->work < state->budget ||
      !llama_dart_cancel_eval_boundary_op(node)) {
    return false;
  }
  state->work = 0.0;
  return true;
}
