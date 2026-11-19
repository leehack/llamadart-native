#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_stream_internal.h"
#include "llama_dart_tts_test_hooks.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

constexpr llama_dart_tts_state kAllStates[] = {
    LLAMA_DART_TTS_STATE_IDLE,       LLAMA_DART_TTS_STATE_PROCESSING_PROMPT,
    LLAMA_DART_TTS_STATE_GENERATING, LLAMA_DART_TTS_STATE_COMPLETED,
    LLAMA_DART_TTS_STATE_CANCELLED,  LLAMA_DART_TTS_STATE_FAILED,
};

void a_streamed_task_is_readable_while_it_generates() {
  assert(!llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_IDLE, true));
  assert(llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_PROCESSING_PROMPT,
                                     true));
  assert(llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_GENERATING, true));
  assert(llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_COMPLETED, true));
  assert(!llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_CANCELLED, true));
  assert(!llama_dart_tts_pcm_readable(LLAMA_DART_TTS_STATE_FAILED, true));
}

void a_plain_task_is_readable_only_when_completed() {
  for (const auto state : kAllStates) {
    assert(llama_dart_tts_pcm_readable(state, false) ==
           (state == LLAMA_DART_TTS_STATE_COMPLETED));
  }
}

void a_valid_delta_is_appended_after_earlier_samples() {
  std::vector<float> pcm = {1.0f};
  const float delta[] = {2.0f, 3.0f};
  assert(llama_dart_tts_append_stream_pcm(
      &pcm, 24000, reinterpret_cast<const char *>(delta), sizeof(delta), 2));
  assert(pcm.size() == 3 && pcm[0] == 1.0f && pcm[1] == 2.0f &&
         pcm[2] == 3.0f);
}

void an_empty_delta_is_accepted() {
  std::vector<float> pcm = {1.0f};
  assert(llama_dart_tts_append_stream_pcm(&pcm, 24000, nullptr, 0, 0));
  assert(pcm.size() == 1);
}

void invalid_metadata_leaves_the_pcm_unchanged() {
  std::vector<float> pcm = {1.0f};
  const float delta[] = {2.0f, 3.0f};
  const char *data = reinterpret_cast<const char *>(delta);
  assert(!llama_dart_tts_append_stream_pcm(&pcm, 24000, data, sizeof(delta),
                                           1));
  assert(!llama_dart_tts_append_stream_pcm(&pcm, 24000, data, sizeof(delta) - 1,
                                           2));
  assert(!llama_dart_tts_append_stream_pcm(&pcm, 0, data, sizeof(delta), 2));
  assert(!llama_dart_tts_append_stream_pcm(&pcm, -1, data, sizeof(delta), 2));
  assert(!llama_dart_tts_append_stream_pcm(&pcm, 24000, data, sizeof(delta),
                                           -2));
  assert(!llama_dart_tts_append_stream_pcm(&pcm, 24000, nullptr, sizeof(delta),
                                           2));
  assert(!llama_dart_tts_append_stream_pcm(
      &pcm, 24000, data, 0, INT64_MAX));
  assert(!llama_dart_tts_append_stream_pcm(nullptr, 24000, data, sizeof(delta),
                                           2));
  assert(pcm.size() == 1 && pcm[0] == 1.0f);
}

void cancel_precedes_flush_failure_and_withdraws_pcm() {
  int32_t status = -1;
  int32_t state = -1;
  int32_t info_status = -1;
  int32_t read_status = -1;

  // Cancel observed after flush fails -> CANCELLED and both reads INVALID_STATE.
  llama_dart_test_tts_finish_streamed_output(-1, true, &status, &state,
                                             &info_status, &read_status);
  assert(status == LLAMA_DART_TTS_STATUS_CANCELLED);
  assert(state == LLAMA_DART_TTS_STATE_CANCELLED);
  assert(info_status == LLAMA_DART_TTS_STATUS_INVALID_STATE);
  assert(read_status == LLAMA_DART_TTS_STATUS_INVALID_STATE);

  // Uncancelled flush failure -> UPSTREAM_ERROR / FAILED and both reads
  // INVALID_STATE.
  llama_dart_test_tts_finish_streamed_output(-1, false, &status, &state,
                                             &info_status, &read_status);
  assert(status == LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR);
  assert(state == LLAMA_DART_TTS_STATE_FAILED);
  assert(info_status == LLAMA_DART_TTS_STATUS_INVALID_STATE);
  assert(read_status == LLAMA_DART_TTS_STATUS_INVALID_STATE);
}

} // namespace

int main() {
  a_streamed_task_is_readable_while_it_generates();
  a_plain_task_is_readable_only_when_completed();
  a_valid_delta_is_appended_after_earlier_samples();
  an_empty_delta_is_accepted();
  invalid_metadata_leaves_the_pcm_unchanged();
  cancel_precedes_flush_failure_and_withdraws_pcm();
  return 0;
}
