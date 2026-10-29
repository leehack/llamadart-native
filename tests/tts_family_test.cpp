#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_tts_family_internal.h"

#include <cassert>
#include <cstring>

namespace {

void test_qwen3_keeps_its_contract() {
  assert(llama_dart_tts_family_from_upstream(MTMD_GEN_AUDIO_TYPE_QWEN3TTS) ==
         LLAMA_DART_TTS_MODEL_TYPE_QWEN3);
  assert(llama_dart_tts_family_runs(LLAMA_DART_TTS_MODEL_TYPE_QWEN3));
  assert(llama_dart_tts_family_primes_code_predictor(
      LLAMA_DART_TTS_MODEL_TYPE_QWEN3));
  assert(llama_dart_tts_family_rejects_request(LLAMA_DART_TTS_MODEL_TYPE_QWEN3,
                                               true, true) == nullptr);
}

void test_unknown_and_absent_generators_never_run() {
  assert(llama_dart_tts_family_from_upstream(MTMD_GEN_AUDIO_TYPE_NONE) ==
         LLAMA_DART_TTS_MODEL_TYPE_NONE);
  assert(llama_dart_tts_family_from_upstream(MTMD_GEN_AUDIO_TYPE_POCKETTTS) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  assert(llama_dart_tts_family_from_upstream(
             static_cast<mtmd_gen_audio_type>(1000)) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
  assert(!llama_dart_tts_family_runs(LLAMA_DART_TTS_MODEL_TYPE_NONE));
  assert(!llama_dart_tts_family_runs(LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN));
  assert(!llama_dart_tts_family_runs(
      static_cast<llama_dart_tts_model_type>(1000)));
}

void test_soprano_runs_only_where_upstream_has_it() {
  assert(llama_dart_tts_family_runs(LLAMA_DART_TTS_MODEL_TYPE_SOPRANO) ==
         static_cast<bool>(LLAMADART_MTMD_HAS_SOPRANO));
  assert(!llama_dart_tts_family_primes_code_predictor(
      LLAMA_DART_TTS_MODEL_TYPE_SOPRANO));
#if LLAMADART_MTMD_HAS_SOPRANO
  assert(llama_dart_tts_family_from_upstream(MTMD_GEN_AUDIO_TYPE_SOPRANO) ==
         LLAMA_DART_TTS_MODEL_TYPE_SOPRANO);
  assert(llama_dart_tts_family_from_upstream(MTMD_GEN_AUDIO_TYPE_LFM3A) ==
         LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN);
#endif
}

void test_soprano_rejects_speaker_and_language() {
  const auto soprano = LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;
  assert(llama_dart_tts_family_rejects_request(soprano, false, false) ==
         nullptr);
  const char *speaker =
      llama_dart_tts_family_rejects_request(soprano, true, false);
  assert(speaker != nullptr && std::strstr(speaker, "speaker") != nullptr);
  const char *language =
      llama_dart_tts_family_rejects_request(soprano, false, true);
  assert(language != nullptr && std::strstr(language, "language") != nullptr);
}

} // namespace

int main() {
  test_qwen3_keeps_its_contract();
  test_unknown_and_absent_generators_never_run();
  test_soprano_runs_only_where_upstream_has_it();
  test_soprano_rejects_speaker_and_language();
  return 0;
}
