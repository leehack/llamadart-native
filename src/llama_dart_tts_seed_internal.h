#pragma once

#include "llama.h"

#include <cstdint>

// LLAMA_DEFAULT_SEED asks for a random seed. One concrete value drives both the
// talker sampler and the code predictor, so a synthesis has a single seed.
template <typename Random>
static inline uint32_t llama_dart_tts_resolve_seed(uint32_t requested,
                                                   Random &&random) {
  uint32_t seed = requested;
  while (seed == LLAMA_DEFAULT_SEED) {
    seed = static_cast<uint32_t>(random());
  }
  return seed;
}

// The code predictor's RNG belongs to the projector and is reseeded only when a
// call's seed differs from the previous call's, so a synthesis that repeats the
// last seed would continue the previous synthesis's stream. One call under this
// seed before the first frame makes that frame reseed the RNG. It is never
// LLAMA_DEFAULT_SEED, which would make the projector read std::random_device.
static inline uint32_t llama_dart_tts_priming_seed(uint32_t seed) {
  return seed != 0 ? seed - 1 : 1;
}
