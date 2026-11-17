#pragma once

#include "llama_dart_wrapper.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// A task's PCM is readable once it completed, and for a streamed task also
// while it generates. CANCELLED, FAILED and IDLE (after a reset) never expose
// PCM, so a caller cannot play audio of a task that did not finish.
static inline bool llama_dart_tts_pcm_readable(llama_dart_tts_state state,
                                               bool streaming) {
  switch (state) {
  case LLAMA_DART_TTS_STATE_COMPLETED:
    return true;
  case LLAMA_DART_TTS_STATE_PROCESSING_PROMPT:
  case LLAMA_DART_TTS_STATE_GENERATING:
    return streaming;
  default:
    return false;
  }
}

// Appends one mtmd_helper_gen_audio_get_output delta. Upstream's buffer is
// valid only until its next call, so the samples are copied here. An empty
// delta is valid: no full window has accumulated yet. Returns false, leaving
// pcm unchanged, for metadata that does not describe float samples.
static inline bool llama_dart_tts_append_stream_pcm(std::vector<float> *pcm,
                                                    int32_t sample_rate,
                                                    const char *data,
                                                    size_t data_len,
                                                    int64_t sample_count) {
  if (pcm == nullptr || sample_rate <= 0 || sample_count < 0 ||
      static_cast<uint64_t>(sample_count) >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      data_len != static_cast<size_t>(sample_count) * sizeof(float) ||
      (data_len > 0 && data == nullptr)) {
    return false;
  }
  if (sample_count == 0) {
    return true;
  }
  const size_t old_size = pcm->size();
  pcm->resize(old_size + static_cast<size_t>(sample_count));
  std::memcpy(pcm->data() + old_size, data, data_len);
  return true;
}
