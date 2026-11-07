#pragma once

#include "llama.h"
#include "llama_dart_wrapper.h"

#include <cmath>
#include <cstdint>
#include <vector>

// The talker's sampler chain. As in upstream common_sampler, the GGUF's
// tokenizer.ggml.suppress_tokens get a -inf bias before any other stage: the
// Qwen3-TTS converter lists the codec's control tokens there, and code2wav
// rejects a control token as an out-of-range code.
static inline llama_sampler *llama_dart_tts_sampler_chain_init(
    const llama_dart_tts_request &request, uint32_t seed, int32_t n_vocab,
    const llama_token *suppress, int32_t n_suppress) {
  llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (sampler == nullptr) {
    return nullptr;
  }
  if (suppress != nullptr && n_suppress > 0) {
    std::vector<llama_logit_bias> bias;
    bias.reserve(static_cast<size_t>(n_suppress));
    for (int32_t i = 0; i < n_suppress; ++i) {
      bias.push_back({suppress[i], -INFINITY});
    }
    llama_sampler_chain_add(
        sampler, llama_sampler_init_logit_bias(
                     n_vocab, static_cast<int32_t>(bias.size()), bias.data()));
  }
  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request.top_k));
  llama_sampler_chain_add(sampler,
                          llama_sampler_init_top_p(request.top_p, 1));
  llama_sampler_chain_add(sampler,
                          llama_sampler_init_min_p(request.min_p, 1));
  llama_sampler_chain_add(
      sampler, llama_sampler_init_temp(request.temperature));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
  return sampler;
}
