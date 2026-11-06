#pragma once

#include "llama_dart_wrapper.h"

#include "mtmd.h"

#include <cctype>
#include <cstdint>
#include <cstddef>

#ifndef LLAMADART_MTMD_HAS_LFM3A_GEN
#define LLAMADART_MTMD_HAS_LFM3A_GEN 0
#endif
#ifndef LLAMADART_MTMD_HAS_SOPRANO
#define LLAMADART_MTMD_HAS_SOPRANO 0
#endif
#ifndef LLAMADART_MTMD_HAS_GEN_VOICES
#define LLAMADART_MTMD_HAS_GEN_VOICES 0
#endif
#ifndef LLAMADART_MTMD_HAS_CHATTERBOX
#define LLAMADART_MTMD_HAS_CHATTERBOX 0
#endif

static constexpr uint32_t
llama_dart_tts_model_type_bit(llama_dart_tts_model_type type) {
  return 1u << static_cast<uint32_t>(type);
}

static constexpr uint32_t llama_dart_tts_build_model_types =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_QWEN3) |
    (LLAMADART_MTMD_HAS_LFM3A_GEN
         ? llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO)
         : 0u) |
    (LLAMADART_MTMD_HAS_SOPRANO
         ? llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO)
         : 0u) |
    // Both select their predefined voices through upstream voice bitmaps.
    (LLAMADART_MTMD_HAS_CHATTERBOX && LLAMADART_MTMD_HAS_GEN_VOICES
         ? llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX)
         : 0u) |
    (LLAMADART_MTMD_HAS_GEN_VOICES
         ? llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS)
         : 0u);

// What llama_dart_tts_get_info and llama_dart_tts_init accepted before the
// _ext exports, and still accept.
static constexpr uint32_t llama_dart_tts_legacy_model_types =
    llama_dart_tts_model_type_bit(LLAMA_DART_TTS_MODEL_TYPE_QWEN3);

static inline llama_dart_tts_model_type
llama_dart_tts_model_type_from_upstream(mtmd_gen_audio_type type) {
  switch (type) {
  case MTMD_GEN_AUDIO_TYPE_NONE:
    return LLAMA_DART_TTS_MODEL_TYPE_NONE;
  case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
    return LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
#if LLAMADART_MTMD_HAS_LFM3A_GEN
  case MTMD_GEN_AUDIO_TYPE_LFM3A:
    return LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;
#endif
#if LLAMADART_MTMD_HAS_SOPRANO
  case MTMD_GEN_AUDIO_TYPE_SOPRANO:
    return LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
#endif
#if LLAMADART_MTMD_HAS_CHATTERBOX && LLAMADART_MTMD_HAS_GEN_VOICES
  case MTMD_GEN_AUDIO_TYPE_CHATTERBOX:
    return LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
#endif
#if LLAMADART_MTMD_HAS_GEN_VOICES
  case MTMD_GEN_AUDIO_TYPE_POCKETTTS:
    return LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS;
#endif
  default:
    return LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN;
  }
}

// A generator the caller did not opt into is UNKNOWN to that caller.
static inline llama_dart_tts_model_type
llama_dart_tts_accepted_model_type(mtmd_gen_audio_type type,
                                   uint32_t accepted_model_types) {
  const llama_dart_tts_model_type model_type =
      llama_dart_tts_model_type_from_upstream(type);
  if (model_type == LLAMA_DART_TTS_MODEL_TYPE_NONE ||
      model_type == LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN) {
    return model_type;
  }
  return (accepted_model_types & llama_dart_tts_model_type_bit(model_type)) != 0
             ? model_type
             : LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN;
}

// What a loaded projector offers, read once by the wrapper.
struct llama_dart_tts_model_traits {
  bool speaker_encoder = false;   // mtmd_support_audio
  bool predefined_voices = false; // mtmd_gen_audio_info.n_voices > 0
  bool multilingual = false;      // Chatterbox Multilingual's position tables
};

static inline uint32_t
llama_dart_tts_model_capabilities(llama_dart_tts_model_type type,
                                  const llama_dart_tts_model_traits &traits) {
  const uint32_t speaker =
      traits.speaker_encoder ? LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE : 0u;
  const uint32_t voice =
      traits.predefined_voices ? LLAMA_DART_TTS_CAPABILITY_VOICE : 0u;
  switch (type) {
  case LLAMA_DART_TTS_MODEL_TYPE_QWEN3:
    return LLAMA_DART_TTS_CAPABILITY_LANGUAGE | speaker;
  case LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO:
    // Its audio encoder is for chat input; the generator has no voice
    // conditioning and ignores a speaker reference. Its voices are names in
    // the "tts" chat template.
    return LLAMA_DART_TTS_CAPABILITY_LANGUAGE |
           LLAMA_DART_TTS_CAPABILITY_VOICE;
  case LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX:
    // Turbo is English only and ignores a language upstream.
    return (traits.multilingual ? LLAMA_DART_TTS_CAPABILITY_LANGUAGE : 0u) |
           speaker | voice;
  case LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS:
    // The language is a property of the weights.
    return speaker | voice;
  default:
    return 0;
  }
}

// An empty language means none was requested, so it is valid for every model.
static inline bool llama_dart_tts_language_supported(uint32_t capabilities,
                                                     const char *language) {
  return language == nullptr || language[0] == '\0' ||
         (capabilities & LLAMA_DART_TTS_CAPABILITY_LANGUAGE) != 0;
}

// Voice names match case-insensitively, as upstream mtmd_gen_audio_get_voice
// and the LFM3-Audio "tts" chat template do.
static inline bool llama_dart_tts_voice_name_equal(const char *name,
                                                   const char *voice) {
  if (name == nullptr || voice == nullptr) {
    return false;
  }
  size_t i = 0;
  for (; name[i] != '\0' && voice[i] != '\0'; ++i) {
    if (std::tolower(static_cast<unsigned char>(name[i])) !=
        std::tolower(static_cast<unsigned char>(voice[i]))) {
      return false;
    }
  }
  return name[i] == voice[i];
}

// LFM3-Audio voices for an mtmd that cannot list them (before upstream
// v0.9.0). Its "tts" chat template falls back to its default voice for any
// other name, so an unknown voice is rejected instead of being ignored.
static inline const char *llama_dart_tts_lfm3_voice_name(int32_t index) {
  static const char *const names[] = {"female", "male"};
  return index >= 0 && index < 2 ? names[index] : nullptr;
}

// Chatterbox and Pocket TTS take a voice as an upstream voice bitmap in place
// of a speaker reference; LFM3-Audio names it in its chat template.
static inline bool
llama_dart_tts_voice_is_conditioning(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX ||
         model_type == LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS;
}

// Upstream Chatterbox fails without a speaker reference or a voice.
static inline bool
llama_dart_tts_requires_conditioning(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
}

// The unconditional sequence that upstream Chatterbox Multilingual decodes
// for classifier-free guidance, half the context's sequences away; -1 when
// the context has no such sequence for seq_id.
static inline int32_t llama_dart_tts_cfg_sequence(uint32_t n_seq_max,
                                                  int32_t seq_id) {
  const int64_t half = static_cast<int64_t>(n_seq_max) / 2;
  if (seq_id < 0 || half == 0 || seq_id >= half) {
    return -1;
  }
  return static_cast<int32_t>(half + seq_id);
}

// Only Qwen3-TTS has a code predictor. Soprano's projector accepts GEN_WAV
// alone and fails a GEN_CODE call.
static inline bool
llama_dart_tts_primes_code_predictor(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
}

// Qwen3-TTS, Soprano and Chatterbox generate each frame from a sampled
// backbone token; LFM3-Audio and Pocket TTS frames come from the hidden state.
static inline bool
llama_dart_tts_samples_backbone_token(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_QWEN3 ||
         model_type == LLAMA_DART_TTS_MODEL_TYPE_SOPRANO ||
         model_type == LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
}

// Upstream Soprano fails a speaker reference and LFM3-Audio ignores one.
// Chatterbox and Pocket TTS fail one without a speaker encoder; Qwen3-TTS
// keeps its decode-time check.
static inline bool
llama_dart_tts_accepts_speaker_reference(llama_dart_tts_model_type model_type,
                                         uint32_t capabilities) {
  switch (model_type) {
  case LLAMA_DART_TTS_MODEL_TYPE_QWEN3:
    return true;
  case LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX:
  case LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS:
    return (capabilities & LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE) != 0;
  default:
    return false;
  }
}
