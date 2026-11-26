#pragma once

#include <algorithm>
#include <cstdint>

// mtmd_helper_gen_audio_step_prompt (llama.cpp v1.1.0) fails above the
// context's n_batch, which a caller can set below the requested prompt batch.
static inline int32_t llama_dart_tts_prompt_batch(int32_t requested,
                                                  uint32_t context_n_batch) {
  const uint32_t limit = std::max<uint32_t>(context_n_batch, 1);
  return static_cast<int32_t>(
      std::min<uint32_t>(static_cast<uint32_t>(requested), limit));
}
