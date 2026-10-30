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
constexpr uint32_t kSoprano =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO);

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

void soprano_follows_the_build_and_the_mask() {
#if LLAMADART_MTMD_HAS_SOPRANO
  assert((llama_dart_tts_build_model_types & kSoprano) != 0);
  assert(llama_dart_tts_model_type_from_upstream(MTMD_GEN_AUDIO_TYPE_SOPRANO) ==
         LLAMA_DART_TTS_MODEL_TYPE_SOPRANO);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_SOPRANO,
                                            kQwen3 | kSoprano) ==
         LLAMA_DART_TTS_MODEL_TYPE_SOPRANO);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_SOPRANO,
                                            llama_dart_tts_legacy_model_types) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#else
  assert((llama_dart_tts_build_model_types & kSoprano) == 0);
#endif
}

void family_behaviour() {
  const auto qwen3 = LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
  const auto lfm3 = LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;
  const auto soprano = LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
  assert(llama_dart_tts_primes_code_predictor(qwen3));
  assert(!llama_dart_tts_primes_code_predictor(lfm3));
  assert(!llama_dart_tts_primes_code_predictor(soprano));
  assert(llama_dart_tts_samples_backbone_token(qwen3));
  assert(!llama_dart_tts_samples_backbone_token(lfm3));
  assert(llama_dart_tts_samples_backbone_token(soprano));
  assert(!llama_dart_tts_rejects_speaker_reference(qwen3));
  assert(llama_dart_tts_rejects_speaker_reference(lfm3));
  assert(llama_dart_tts_rejects_speaker_reference(soprano));
  assert(llama_dart_tts_voice_supported(soprano, nullptr));
  assert(!llama_dart_tts_voice_supported(soprano, "female"));
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

void capabilities_are_exact_per_model_type() {
  constexpr uint32_t kLanguage = LLAMA_DART_TTS_CAPABILITY_LANGUAGE;
  constexpr uint32_t kSpeaker = LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE;
  constexpr uint32_t kVoice = LLAMA_DART_TTS_CAPABILITY_VOICE;
  const auto qwen3 = LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
  assert(llama_dart_tts_model_capabilities(qwen3, true) == (kLanguage | kSpeaker));
  assert(llama_dart_tts_model_capabilities(qwen3, false) == kLanguage);
  // A voice is not selectable on Qwen3-TTS, so start_with_voice is UNSUPPORTED.
  assert((llama_dart_tts_model_capabilities(qwen3, true) & kVoice) == 0);
  assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO,
                                           true) == (kLanguage | kVoice));
  assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO,
                                           true) == 0);
  assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_NONE,
                                           true) == 0);
  assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN,
                                           true) == 0);
}

void a_generator_the_caller_did_not_accept_has_no_capabilities() {
  // Legacy llama_dart_tts_get_info on an LFM3 or Soprano model.
  for (int upstream = MTMD_GEN_AUDIO_TYPE_QWEN3TTS + 1; upstream < 64;
       ++upstream) {
    const auto type = llama_dart_tts_accepted_model_type(
        static_cast<mtmd_gen_audio_type>(upstream),
        llama_dart_tts_legacy_model_types);
    assert(type == LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
    assert(llama_dart_tts_model_capabilities(type, true) == 0);
  }
}

void language_is_checked_against_the_capability() {
  constexpr uint32_t kLanguage = LLAMA_DART_TTS_CAPABILITY_LANGUAGE;
  // Soprano takes no language, and an empty one means none was requested.
  assert(llama_dart_tts_language_supported(0, nullptr));
  assert(llama_dart_tts_language_supported(0, ""));
  assert(!llama_dart_tts_language_supported(0, "en"));
  assert(llama_dart_tts_language_supported(kLanguage, "en"));
  assert(llama_dart_tts_language_supported(kLanguage, ""));
}

} // namespace

int main() {
  legacy_mask_keeps_the_qwen3_only_contract();
  a_type_outside_the_mask_is_unknown();
  lfm3_audio_follows_the_build();
  soprano_follows_the_build_and_the_mask();
  family_behaviour();
  voices();
  capabilities_are_exact_per_model_type();
  a_generator_the_caller_did_not_accept_has_no_capabilities();
  language_is_checked_against_the_capability();
  return 0;
}
