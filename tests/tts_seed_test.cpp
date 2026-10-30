#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_seed_internal.h"

#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

namespace {

constexpr uint32_t kEdgeSeeds[] = {0u, 1u, 0x7fffffffu, LLAMA_DEFAULT_SEED - 1};

// The projector RNG rule of clip_image_batch_encode at llama.cpp v0.7.0, the
// last release without mtmd_gen_state (LLAMADART_MTMD_HAS_GEN_STATE false),
// which is the only build that primes the code predictor:
// https://github.com/ggml-org/llama.cpp/blob/620d2225cc436ab0b001e3200e590e3b286ef9ab/tools/mtmd/clip.cpp#L4603-L4606
struct projector_rng {
  std::mt19937 rng{12345u};
  uint32_t rng_seed = LLAMA_DEFAULT_SEED;

  uint32_t encode(uint32_t seed) {
    if (seed != rng_seed) {
      rng_seed = seed;
      assert(seed != LLAMA_DEFAULT_SEED);
      rng.seed(seed);
    }
    return rng();
  }
};

std::vector<uint32_t> synthesize(projector_rng *projector, uint32_t seed,
                                 bool prime, int frames) {
  if (prime) {
    projector->encode(llama_dart_tts_priming_seed(seed));
  }
  std::vector<uint32_t> draws;
  for (int i = 0; i < frames; ++i) {
    draws.push_back(projector->encode(seed));
  }
  return draws;
}

std::vector<uint32_t> fresh_draws(uint32_t seed, int frames) {
  std::mt19937 rng(seed);
  std::vector<uint32_t> draws;
  for (int i = 0; i < frames; ++i) {
    draws.push_back(rng());
  }
  return draws;
}

void test_explicit_seeds_pass_through() {
  for (uint32_t seed : kEdgeSeeds) {
    bool drew = false;
    assert(llama_dart_tts_resolve_seed(seed, [&] {
             drew = true;
             return 7u;
           }) == seed);
    assert(!drew);
  }
}

void test_default_seed_draws_a_concrete_seed() {
  std::vector<uint32_t> values = {LLAMA_DEFAULT_SEED, LLAMA_DEFAULT_SEED, 42u};
  size_t next = 0;
  assert(llama_dart_tts_resolve_seed(LLAMA_DEFAULT_SEED,
                                     [&] { return values[next++]; }) == 42u);
  assert(next == values.size());
}

void test_priming_seed_forces_a_reseed_without_randomness() {
  for (uint32_t seed : kEdgeSeeds) {
    const uint32_t priming = llama_dart_tts_priming_seed(seed);
    assert(priming != seed);
    assert(priming != LLAMA_DEFAULT_SEED);
  }
}

void test_repeated_seed_restarts_the_projector_rng() {
  for (uint32_t seed : kEdgeSeeds) {
    projector_rng unprimed;
    const std::vector<uint32_t> first = synthesize(&unprimed, seed, false, 4);
    assert(first == fresh_draws(seed, 4));
    assert(synthesize(&unprimed, seed, false, 4) != first);

    projector_rng primed;
    synthesize(&primed, seed, false, 3);
    synthesize(&primed, seed ^ 0x10u, true, 5);
    for (int run = 0; run < 3; ++run) {
      assert(synthesize(&primed, seed, true, 4) == fresh_draws(seed, 4));
    }
  }
}

} // namespace

int main() {
  test_explicit_seeds_pass_through();
  test_default_seed_draws_a_concrete_seed();
  test_priming_seed_forces_a_reseed_without_randomness();
  test_repeated_seed_restarts_the_projector_rng();
  return 0;
}
