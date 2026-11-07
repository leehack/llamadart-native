#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_types_internal.h"

#include <cassert>
#include <cstdint>
#include <cstring>

namespace {

constexpr uint32_t kQwen3 =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_QWEN3);
constexpr uint32_t kLfm3 =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO);
constexpr uint32_t kSoprano =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO);
constexpr uint32_t kChatterbox =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX);
constexpr uint32_t kPocket =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS);

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

void chatterbox_and_pocket_follow_the_build_and_the_mask() {
#if LLAMADART_MTMD_HAS_CHATTERBOX && LLAMADART_MTMD_HAS_GEN_VOICES
  assert((llama_dart_tts_build_model_types & kChatterbox) != 0);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_CHATTERBOX,
                                            kQwen3 | kChatterbox) ==
         LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_CHATTERBOX,
                                            kQwen3 | kLfm3 | kSoprano |
                                                kPocket) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#else
  assert((llama_dart_tts_build_model_types & kChatterbox) == 0);
#endif
#if LLAMADART_MTMD_HAS_GEN_VOICES
  assert((llama_dart_tts_build_model_types & kPocket) != 0);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_POCKETTTS,
                                            kQwen3 | kPocket) ==
         LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_POCKETTTS,
                                            kQwen3 | kLfm3 | kSoprano |
                                                kChatterbox) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#else
  // Without voice bitmaps Pocket TTS keeps the refusal it had before.
  assert((llama_dart_tts_build_model_types & kPocket) == 0);
  assert(llama_dart_tts_accepted_model_type(MTMD_GEN_AUDIO_TYPE_POCKETTTS,
                                            ~0u) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#endif
}

void family_behaviour() {
  const auto qwen3 = LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
  const auto lfm3 = LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;
  const auto soprano = LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
  const auto chatterbox = LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
  const auto pocket = LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS;
  constexpr uint32_t kSpeaker = LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE;
  assert(llama_dart_tts_primes_code_predictor(qwen3));
  for (const auto type : {lfm3, soprano, chatterbox, pocket}) {
    assert(!llama_dart_tts_primes_code_predictor(type));
  }
  assert(llama_dart_tts_samples_backbone_token(qwen3));
  assert(!llama_dart_tts_samples_backbone_token(lfm3));
  assert(llama_dart_tts_samples_backbone_token(soprano));
  assert(llama_dart_tts_samples_backbone_token(chatterbox));
  assert(!llama_dart_tts_samples_backbone_token(pocket));
  // Qwen3-TTS keeps failing an undecodable reference at decode time.
  assert(llama_dart_tts_accepts_speaker_reference(qwen3, 0));
  assert(!llama_dart_tts_accepts_speaker_reference(lfm3, kSpeaker));
  assert(!llama_dart_tts_accepts_speaker_reference(soprano, kSpeaker));
  for (const auto type : {chatterbox, pocket}) {
    assert(llama_dart_tts_accepts_speaker_reference(type, kSpeaker));
    assert(!llama_dart_tts_accepts_speaker_reference(type, 0));
  }
  assert(!llama_dart_tts_accepts_speaker_reference(
      LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN, kSpeaker));
  assert(llama_dart_tts_voice_is_conditioning(chatterbox));
  assert(llama_dart_tts_voice_is_conditioning(pocket));
  for (const auto type : {qwen3, lfm3, soprano}) {
    assert(!llama_dart_tts_voice_is_conditioning(type));
  }
  assert(llama_dart_tts_requires_conditioning(chatterbox));
  for (const auto type : {qwen3, lfm3, soprano, pocket}) {
    assert(!llama_dart_tts_requires_conditioning(type));
  }
}

void voices() {
  assert(llama_dart_tts_voice_name_equal("female", "female"));
  assert(llama_dart_tts_voice_name_equal("female", "Female"));
  assert(llama_dart_tts_voice_name_equal("male", "MALE"));
  assert(llama_dart_tts_voice_name_equal("bill_boerst", "Bill_Boerst"));
  assert(!llama_dart_tts_voice_name_equal("female", ""));
  assert(!llama_dart_tts_voice_name_equal("male", "mal"));
  assert(!llama_dart_tts_voice_name_equal("male", "males"));
  assert(!llama_dart_tts_voice_name_equal("female", "femalee"));
  assert(!llama_dart_tts_voice_name_equal("female", "child"));
  assert(!llama_dart_tts_voice_name_equal(nullptr, "male"));
  assert(!llama_dart_tts_voice_name_equal("male", nullptr));
  assert(std::strcmp(llama_dart_tts_lfm3_voice_name(0), "female") == 0);
  assert(std::strcmp(llama_dart_tts_lfm3_voice_name(1), "male") == 0);
  assert(llama_dart_tts_lfm3_voice_name(2) == nullptr);
  assert(llama_dart_tts_lfm3_voice_name(-1) == nullptr);
}

void cfg_sequence_is_half_the_context_away() {
  // Upstream pairs sequence s with n_seq_max / 2 + s.
  assert(llama_dart_tts_cfg_sequence(2, 0) == 1);
  assert(llama_dart_tts_cfg_sequence(4, 1) == 3);
  assert(llama_dart_tts_cfg_sequence(5, 1) == 3);
  assert(llama_dart_tts_cfg_sequence(4, 2) == -1);
  assert(llama_dart_tts_cfg_sequence(5, 2) == -1);
  assert(llama_dart_tts_cfg_sequence(1, 0) == -1);
  assert(llama_dart_tts_cfg_sequence(0, 0) == -1);
  assert(llama_dart_tts_cfg_sequence(4, -1) == -1);
  assert(llama_dart_tts_cfg_sequence(UINT32_MAX, INT32_MAX) == -1);
}

void capabilities_are_exact_per_model_type() {
  constexpr uint32_t kLanguage = LLAMA_DART_TTS_CAPABILITY_LANGUAGE;
  constexpr uint32_t kSpeaker = LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE;
  constexpr uint32_t kVoice = LLAMA_DART_TTS_CAPABILITY_VOICE;
  llama_dart_tts_model_traits none;
  llama_dart_tts_model_traits all;
  all.speaker_encoder = true;
  all.predefined_voices = true;
  all.multilingual = true;
  llama_dart_tts_model_traits encoder;
  encoder.speaker_encoder = true;
  llama_dart_tts_model_traits voices;
  voices.predefined_voices = true;
  const auto qwen3 = LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
  assert(llama_dart_tts_model_capabilities(qwen3, encoder) ==
         (kLanguage | kSpeaker));
  assert(llama_dart_tts_model_capabilities(qwen3, none) == kLanguage);
  // A voice is not selectable on Qwen3-TTS, so start_with_voice is UNSUPPORTED.
  assert((llama_dart_tts_model_capabilities(qwen3, all) & kVoice) == 0);
  for (const auto &traits : {none, all}) {
    assert(llama_dart_tts_model_capabilities(
               LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO, traits) ==
           (kLanguage | kVoice));
    assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO,
                                             traits) == 0);
    assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_NONE,
                                             traits) == 0);
    assert(llama_dart_tts_model_capabilities(LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN,
                                             traits) == 0);
  }
  const auto chatterbox = LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
  assert(llama_dart_tts_model_capabilities(chatterbox, all) ==
         (kLanguage | kSpeaker | kVoice));
  // Turbo: a speaker encoder and the shipped voice, no language.
  llama_dart_tts_model_traits turbo = all;
  turbo.multilingual = false;
  assert(llama_dart_tts_model_capabilities(chatterbox, turbo) ==
         (kSpeaker | kVoice));
  assert(llama_dart_tts_model_capabilities(chatterbox, encoder) == kSpeaker);
  assert(llama_dart_tts_model_capabilities(chatterbox, voices) == kVoice);
  const auto pocket = LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS;
  // The language of Pocket TTS is that of its weights.
  assert(llama_dart_tts_model_capabilities(pocket, all) == (kSpeaker | kVoice));
  assert(llama_dart_tts_model_capabilities(pocket, voices) == kVoice);
  assert(llama_dart_tts_model_capabilities(pocket, encoder) == kSpeaker);
  assert(llama_dart_tts_model_capabilities(pocket, none) == 0);
}

void a_generator_the_caller_did_not_accept_has_no_capabilities() {
  // Legacy llama_dart_tts_get_info on any generator but Qwen3-TTS.
  for (int upstream = MTMD_GEN_AUDIO_TYPE_QWEN3TTS + 1; upstream < 64;
       ++upstream) {
    const auto type = llama_dart_tts_accepted_model_type(
        static_cast<mtmd_gen_audio_type>(upstream),
        llama_dart_tts_legacy_model_types);
    assert(type == LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
    llama_dart_tts_model_traits all;
    all.speaker_encoder = true;
    all.predefined_voices = true;
    all.multilingual = true;
    assert(llama_dart_tts_model_capabilities(type, all) == 0);
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
  chatterbox_and_pocket_follow_the_build_and_the_mask();
  family_behaviour();
  voices();
  cfg_sequence_is_half_the_context_away();
  capabilities_are_exact_per_model_type();
  a_generator_the_caller_did_not_accept_has_no_capabilities();
  language_is_checked_against_the_capability();
  return 0;
}
