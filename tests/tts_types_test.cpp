#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_types_internal.h"

#include <cassert>
#include <cstdint>

namespace {

constexpr uint32_t kQwen3 =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_QWEN3);
constexpr uint32_t kLfm3 =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO);

void legacy_mask_keeps_the_qwen3_only_contract() {
  assert(llama_dart_tts_legacy_model_types == kQwen3);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_NONE,
                                            kQwen3) ==
         LLAMA_DART_TTS_MODEL_TYPE_NONE);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_QWEN3TTS,
                                            kQwen3) ==
         LLAMA_DART_TTS_MODEL_TYPE_QWEN3);
  // Upstream generator values the wrapper has no type for.
  for (int upstream = MTMD_GEN_AUDIO_TYPE_QWEN3TTS + 1; upstream < 64;
       ++upstream) {
    const auto type = static_cast<mtmd_gen_audio_type>(upstream);
    assert(llama_dart_tts_accepted_model_type(type, kQwen3) ==
           LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  }
}

void a_type_outside_the_mask_is_unknown() {
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_QWEN3TTS,
                                            kLfm3) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_QWEN3TTS, 0) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_NONE, 0) ==
         LLAMA_DART_TTS_MODEL_TYPE_NONE);
}

void lfm3_audio_follows_the_build() {
  assert((llama_dart_tts_build_model_types & kQwen3) != 0);
#if LLAMADART_MTMD_HAS_LFM3A_GEN
  assert((llama_dart_tts_build_model_types & kLfm3) != 0);
  assert(llama_dart_tts_model_type_from_upstream(MTMD_GEN_AUDIO_TYPE_LFM3A) ==
         LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_LFM3A,
                                            kQwen3 | kLfm3) ==
         LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_LFM3A,
                                            kQwen3) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  // Soprano and Pocket TTS stay unknown to an LFM3 caller.
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_SOPRANO,
                                            kQwen3 | kLfm3) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_POCKETTTS,
                                            kQwen3 | kLfm3) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#else
  assert((llama_dart_tts_build_model_types & kLfm3) == 0);
#endif
}

void voices() {
  const auto lfm3 = LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;
  assert(llama_dart_tts_voice_supported(lfm3, nullptr));
  assert(llama_dart_tts_voice_supported(lfm3, "female"));
  assert(llama_dart_tts_voice_supported(lfm3, "male"));
  assert(llama_dart_tts_voice_supported(lfm3, "MALE"));
  assert(llama_dart_tts_voice_supported(lfm3, "Female"));
  assert(!llama_dart_tts_voice_supported(lfm3, ""));
  assert(!llama_dart_tts_voice_supported(lfm3, "mal"));
  assert(!llama_dart_tts_voice_supported(lfm3, "males"));
  assert(!llama_dart_tts_voice_supported(lfm3, "femalee"));
  assert(!llama_dart_tts_voice_supported(lfm3, "child"));
  assert(llama_dart_tts_voice_supported(LLAMA_DART_TTS_MODEL_TYPE_QWEN3,
                                        nullptr));
  assert(!llama_dart_tts_voice_supported(LLAMA_DART_TTS_MODEL_TYPE_QWEN3,
                                         "male"));
}

} // namespace

int main() {
  legacy_mask_keeps_the_qwen3_only_contract();
  a_type_outside_the_mask_is_unknown();
  lfm3_audio_follows_the_build();
  voices();
  return 0;
}
