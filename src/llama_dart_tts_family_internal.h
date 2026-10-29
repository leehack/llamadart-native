#pragma once

#include "llama_dart_wrapper.h"

#include "mtmd.h"

// The families llama_dart_tts_get_info_v2 and llama_dart_tts_init_v2 run.
// llama_dart_tts_get_info and llama_dart_tts_init keep their Qwen3-only
// mapping in the wrapper.
static inline llama_dart_tts_model_type
llama_dart_tts_family_from_upstream(mtmd_gen_audio_type type) {
  switch (type) {
  case MTMD_GEN_AUDIO_TYPE_NONE:
    return LLAMA_DART_TTS_MODEL_TYPE_NONE;
  case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
    return LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
#if LLAMADART_MTMD_HAS_SOPRANO
  case MTMD_GEN_AUDIO_TYPE_SOPRANO:
    return LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
#endif
  default:
    return LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN;
  }
}

static inline bool llama_dart_tts_family_runs(int32_t model_type) {
  switch (model_type) {
  case LLAMA_DART_TTS_MODEL_TYPE_QWEN3:
    return true;
  case LLAMA_DART_TTS_MODEL_TYPE_SOPRANO:
    return LLAMADART_MTMD_HAS_SOPRANO != 0;
  default:
    return false;
  }
}

// Only Qwen3-TTS has a code predictor. Soprano's projector accepts GEN_WAV
// alone and fails a GEN_CODE call.
static inline bool
llama_dart_tts_family_primes_code_predictor(llama_dart_tts_model_type type) {
  return type == LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
}

// Returns why the family cannot run a request, or nullptr. Upstream Soprano
// rejects a speaker reference and ignores a language.
static inline const char *
llama_dart_tts_family_rejects_request(llama_dart_tts_model_type type,
                                      bool has_speaker_audio,
                                      bool has_language) {
  if (type != LLAMA_DART_TTS_MODEL_TYPE_SOPRANO) {
    return nullptr;
  }
  if (has_speaker_audio) {
    return "Soprano does not accept a speaker reference";
  }
  if (has_language) {
    return "Soprano does not accept a language";
  }
  return nullptr;
}
