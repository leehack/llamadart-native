#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_batch_internal.h"

#include <cassert>
#include <cstdint>

int main() {
  // A context that clamps n_batch below the request must not see more.
  assert(llama_dart_tts_prompt_batch(512, 256) == 256);
  assert(llama_dart_tts_prompt_batch(512, 1) == 1);
  assert(llama_dart_tts_prompt_batch(512, 511) == 511);
  // At or above the request, the request is kept.
  assert(llama_dart_tts_prompt_batch(512, 512) == 512);
  assert(llama_dart_tts_prompt_batch(512, 2048) == 512);
  assert(llama_dart_tts_prompt_batch(1, 2048) == 1);
  // Never zero or negative for a usable context.
  assert(llama_dart_tts_prompt_batch(512, 0) == 1);
  assert(llama_dart_tts_prompt_batch(512, UINT32_MAX) == 512);
  return 0;
}
