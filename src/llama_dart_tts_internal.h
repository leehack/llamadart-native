#pragma once

#include "llama_dart_barrier_internal.h"
#include "llama_dart_wrapper.h"

#include "mtmd-helper.h"
#include "mtmd.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

struct llama_dart_tts {
  llama_context *llama = nullptr;
  mtmd_context *mtmd = nullptr;
  mtmd_helper_gen_audio *generator = nullptr;
  llama_sampler *sampler = nullptr;
  mtmd_bitmap *speaker = nullptr;
  std::atomic<bool> cancel_requested{false};
  const int8_t *cancel_flag = nullptr;
  llama_dart_tts_state state = LLAMA_DART_TTS_STATE_IDLE;
  llama_seq_id sequence_id = 0;
  bool owns_sequence = false;
  int32_t prompt_batch_size = 512;
  int32_t max_frames = 512;
  int32_t prompt_tokens_remaining = 0;
  int32_t frames_generated = 0;
  bool truncated = false;
  int32_t sample_rate = 0;
  int64_t sample_count = 0;
  std::vector<float> pcm;
  std::string language;
  std::string error;
};

static inline void
llama_dart_tts_release_task_resources(llama_dart_tts *tts);

static inline llama_dart_tts_status llama_dart_tts_fail(
    llama_dart_tts *tts, llama_dart_tts_status status, const char *message) {
  if (tts != nullptr) {
    tts->state = LLAMA_DART_TTS_STATE_FAILED;
    tts->error = message != nullptr ? message : "unknown TTS error";
    llama_dart_tts_release_task_resources(tts);
  }
  return status;
}

static inline llama_dart_tts_status llama_dart_tts_error(
    llama_dart_tts *tts, llama_dart_tts_status status, const char *message) {
  if (tts != nullptr) {
    tts->error = message != nullptr ? message : "unknown TTS error";
  }
  return status;
}

static inline void
llama_dart_tts_release_task_resources(llama_dart_tts *tts) {
  if (tts == nullptr) {
    return;
  }
  if (tts->speaker != nullptr) {
    mtmd_bitmap_free(tts->speaker);
    tts->speaker = nullptr;
  }
  if (tts->sampler != nullptr) {
    llama_sampler_free(tts->sampler);
    tts->sampler = nullptr;
  }
  if (tts->generator != nullptr) {
    mtmd_helper_gen_audio_reset(tts->generator);
  }
  if (tts->owns_sequence) {
    llama_memory_seq_rm(llama_get_memory(tts->llama), tts->sequence_id, 0, -1);
    tts->owns_sequence = false;
  }
  tts->cancel_flag = nullptr;
}

// A TTS call behind the barrier. An exception fails the task like any other
// upstream error, so that the next step does not continue it.
template <typename Call>
static llama_dart_tts_status llama_dart_tts_barrier(llama_dart_tts *tts,
                                                    Call &&call) noexcept {
  const llama_dart_tts_status status =
      llama_dart_barrier(LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR, call);
  if (tts != nullptr && llama_dart_error.set) {
    // Releasing the task's resources calls llama.cpp again.
    llama_dart_exit_call_begin();
    try {
      llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                          llama_dart_error.message);
    } catch (...) {
      tts->state = LLAMA_DART_TTS_STATE_FAILED;
    }
    llama_dart_exit_call_end();
  }
  return status;
}
