#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_sampler_internal.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace {

// A Qwen3-TTS-like tail: the codec's control tokens follow its codes, and the
// GGUF suppresses every control token except the end-of-speech one.
constexpr int32_t kVocab = 64;
constexpr int32_t kFirstControl = 48;
constexpr llama_token kEndOfSpeech = 50;

std::vector<llama_token> control_tokens() {
  std::vector<llama_token> tokens;
  for (llama_token token = kFirstControl; token < kVocab; ++token) {
    if (token != kEndOfSpeech) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

bool is_control(llama_token token) {
  return token >= kFirstControl && token != kEndOfSpeech;
}

llama_dart_tts_request request_with_top_k(int32_t top_k) {
  llama_dart_tts_request request{};
  request.top_k = top_k;
  request.top_p = 0.95f;
  request.min_p = 0.0f;
  request.temperature = 0.8f;
  return request;
}

// Every control token outranks every code.
llama_token sample(const llama_dart_tts_request &request, uint32_t seed,
                   const std::vector<llama_token> &suppress) {
  llama_sampler *sampler = llama_dart_tts_sampler_chain_init(
      request, seed, kVocab, suppress.empty() ? nullptr : suppress.data(),
      static_cast<int32_t>(suppress.size()));
  assert(sampler != nullptr);
  std::vector<llama_token_data> data;
  for (llama_token token = 0; token < kVocab; ++token) {
    const float logit = is_control(token) ? 10.0f + token : token * 0.01f;
    data.push_back({token, logit, 0.0f});
  }
  llama_token_data_array candidates{data.data(), data.size(), -1, false};
  llama_sampler_apply(sampler, &candidates);
  assert(candidates.selected >= 0 &&
         static_cast<size_t>(candidates.selected) < candidates.size);
  const llama_token token = candidates.data[candidates.selected].id;
  llama_sampler_free(sampler);
  return token;
}

void test_suppressed_tokens_are_never_sampled() {
  const std::vector<llama_token> suppress = control_tokens();
  for (int32_t top_k : {0, 1, 40, kVocab}) {
    for (uint32_t seed = 1; seed <= 64; ++seed) {
      assert(!is_control(sample(request_with_top_k(top_k), seed, suppress)));
    }
  }
  assert(sample(request_with_top_k(1), 1, suppress) == kEndOfSpeech);
}

void test_without_suppress_tokens_the_chain_is_unbiased() {
  assert(sample(request_with_top_k(1), 1, {}) == kVocab - 1);
}

}  // namespace

int main() {
  test_suppressed_tokens_are_never_sampled();
  test_without_suppress_tokens_the_chain_is_unbiased();
  return 0;
}
