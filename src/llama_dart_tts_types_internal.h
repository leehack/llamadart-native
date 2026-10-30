#pragma once

#include "llama_dart_wrapper.h"

#include "mtmd.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#ifndef LLAMADART_MTMD_HAS_LFM3A_GEN
#define LLAMADART_MTMD_HAS_LFM3A_GEN 0
#endif
#ifndef LLAMADART_MTMD_HAS_SOPRANO
#define LLAMADART_MTMD_HAS_SOPRANO 0
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

static inline uint32_t
llama_dart_tts_model_capabilities(llama_dart_tts_model_type type,
                                  bool takes_speaker_audio) {
  switch (type) {
  case LLAMA_DART_TTS_MODEL_TYPE_QWEN3:
    return LLAMA_DART_TTS_CAPABILITY_LANGUAGE |
           (takes_speaker_audio ? LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE
                                : 0u);
  case LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO:
    // Its audio encoder is for chat input; the generator has no voice
    // conditioning and ignores a speaker reference.
    return LLAMA_DART_TTS_CAPABILITY_LANGUAGE |
           LLAMA_DART_TTS_CAPABILITY_VOICE;
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

// The LFM3-Audio "tts" chat template falls back to its default voice for any
// other name, so an unknown voice is rejected here instead of being ignored.
static inline bool llama_dart_tts_voice_supported(
    llama_dart_tts_model_type model_type, const char *voice) {
  if (voice == nullptr) {
    return true;
  }
  if (model_type != LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO) {
    return false;
  }
  for (const char *name : {"female", "male"}) {
    const size_t length = std::strlen(name);
    if (std::strlen(voice) != length) {
      continue;
    }
    size_t i = 0;
    while (i < length &&
           std::tolower(static_cast<unsigned char>(voice[i])) == name[i]) {
      ++i;
    }
    if (i == length) {
      return true;
    }
  }
  return false;
}

// Only Qwen3-TTS has a code predictor. Soprano's projector accepts GEN_WAV
// alone and fails a GEN_CODE call.
static inline bool
llama_dart_tts_primes_code_predictor(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
}

// Qwen3-TTS and Soprano generate each frame from a sampled backbone token;
// LFM3-Audio frames come from the hidden state alone.
static inline bool
llama_dart_tts_samples_backbone_token(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_QWEN3 ||
         model_type == LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
}

// Upstream Soprano fails a speaker reference; LFM3-Audio ignores one.
static inline bool
llama_dart_tts_rejects_speaker_reference(llama_dart_tts_model_type model_type) {
  return model_type == LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO ||
         model_type == LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
}
