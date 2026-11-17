#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_asr_stream_internal.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr size_t kRate = llama_dart_asr_stream_sample_rate;

void classification_follows_the_two_upstream_routes() {
  using mode = llama_dart_asr_stream_mode;
  assert(llama_dart_asr_stream_classify(true, false, false) ==
         LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY);
  assert(llama_dart_asr_stream_classify(true, false, true) ==
         LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY);
  assert(llama_dart_asr_stream_classify(true, true, true) ==
         LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS);
  // A stream that only prefills a chat turn (Gemma 4, LFM3-Omni) writes no text.
  assert(llama_dart_asr_stream_classify(true, true, false) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);
  assert(llama_dart_asr_stream_classify(false, false, false) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);
  assert(llama_dart_asr_stream_classify(false, true, true) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);

  const uint32_t token_delay =
      llama_dart_asr_stream_mode_bit(LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY);
  const uint32_t prompted = llama_dart_asr_stream_mode_bit(
      LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS);
  assert(llama_dart_asr_stream_accepted_mode(
             LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY, token_delay) ==
         LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY);
  assert(llama_dart_asr_stream_accepted_mode(
             LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY, prompted) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);
  assert(llama_dart_asr_stream_accepted_mode(
             LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS, 0) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);
  assert(llama_dart_asr_stream_accepted_mode(
             LLAMA_DART_ASR_STREAM_MODE_NONE, ~0u) ==
         LLAMA_DART_ASR_STREAM_MODE_NONE);
  (void)sizeof(mode);
}

void sizes_follow_the_upstream_streams() {
  // Voxtral Realtime with 6 delay tokens: BOS, 32 left pad rows, the delay.
  assert(llama_dart_asr_stream_required_batch(
             LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY, 0, 6) == 39);
  assert(llama_dart_asr_stream_required_batch(
             LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS, 20, 0) == 33);
  // 4096 positions leave about 5.4 minutes after the 39 prefix rows.
  assert(llama_dart_asr_stream_max_session_ms(4096, 39) == (4096 - 39) * 80);
  assert(llama_dart_asr_stream_max_session_ms(39, 39) == 0);
  assert(llama_dart_asr_stream_max_session_ms(10, 39) == 0);
  assert(llama_dart_asr_stream_step_samples(
             LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY) == kRate * 320 / 1000);
  assert(llama_dart_asr_stream_step_samples(
             LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS) == kRate * 960 / 1000);
}

void the_meter_accepts_exactly_up_to_its_bound() {
  llama_dart_asr_input_meter meter(1000, 100);
  const std::vector<float> audio(2000, 0.5f);
  assert(meter.accept(audio.data(), 600) == 600);
  assert(meter.buffered() == 600);
  assert(meter.accept(audio.data(), 600) == 400);
  assert(meter.buffered() == 1000);
  assert(meter.accept(audio.data(), 1) == 0);

  const float *chunk = nullptr;
  assert(meter.next_chunk(&chunk) == 100);
  meter.consume(100);
  assert(meter.buffered() == 900);
  assert(meter.accept(audio.data(), 500) == 100);
  assert(meter.buffered() == 1000);

  meter.clear();
  assert(meter.buffered() == 0);
  assert(meter.accept(audio.data(), 1000) == 1000);
}

void the_meter_never_hands_out_more_than_one_step() {
  // 297 s pushed as fast as the bound allows, as the smoke does.
  const size_t step = kRate * 320 / 1000;
  llama_dart_asr_input_meter meter(kRate * 30, step);
  std::vector<float> audio(kRate * 297);
  for (size_t i = 0; i < audio.size(); ++i) {
    audio[i] = static_cast<float>(i);
  }
  size_t offered = 0;
  size_t consumed = 0;
  size_t largest = 0;
  while (consumed < audio.size()) {
    offered += meter.accept(audio.data() + offered, audio.size() - offered);
    const float *chunk = nullptr;
    const size_t n = meter.next_chunk(&chunk);
    assert(n > 0 && n <= step);
    largest = std::max(largest, n);
    for (size_t i = 0; i < n; ++i) {
      assert(chunk[i] == static_cast<float>(consumed + i));
    }
    meter.consume(n);
    consumed += n;
  }
  assert(largest == step);
  assert(meter.buffered() == 0);
}

void the_text_buffer_never_splits_a_code_point() {
  llama_dart_asr_text_buffer text;
  // U+20AC in three bytes, as two tokens: the first reads nothing.
  text.append("a\xE2\x82", 3);
  assert(text.available() == 1);
  text.append("\xAC" "b", 2);
  assert(text.available() == 5);

  char out[8];
  // A capacity inside the euro sign stops before it.
  assert(text.read(out, 3) == 1);
  assert(out[0] == 'a');
  assert(text.read(out, 2) == 0);
  assert(text.available() == 4);
  assert(text.read(out, sizeof(out)) == 4);
  assert(std::memcmp(out, "\xE2\x82\xAC" "b", 4) == 0);
  assert(text.available() == 0);
}

void the_text_buffer_reads_one_byte_pieces() {
  llama_dart_asr_text_buffer text;
  const std::string sentence = "caf\xC3\xA9 \xF0\x9F\x98\x80 ok";
  std::string read;
  for (char byte : sentence) {
    text.append(&byte, 1);
    char out[4];
    const size_t n = text.read(out, sizeof(out));
    read.append(out, n);
  }
  assert(read == sentence);
}

void invalid_and_unfinished_bytes_become_replacement_characters() {
  llama_dart_asr_text_buffer text;
  char out[16];
  // A continuation byte with no lead, and a lead followed by a non-continuation.
  text.append("\x80" "a\xE2" "b", 4);
  assert(text.read(out, sizeof(out)) == 3 + 1 + 3 + 1);
  assert(std::memcmp(out, "\xEF\xBF\xBD" "a\xEF\xBF\xBD" "b", 8) == 0);

  // An overlong lead and an out-of-range second byte.
  text.append("\xC0\xAF\xE0\x80\x80", 5);
  assert(text.available() == 5 * 3);
  text.clear();

  // The stream ends inside a sequence.
  text.append("x\xF0\x9F", 3);
  assert(text.available() == 1);
  text.finish();
  assert(text.available() == 1 + 3);
  assert(text.read(out, sizeof(out)) == 4);
  assert(std::memcmp(out, "x\xEF\xBF\xBD", 4) == 0);
  text.finish();
  assert(text.available() == 0);
}

} // namespace

int main() {
  classification_follows_the_two_upstream_routes();
  sizes_follow_the_upstream_streams();
  the_meter_accepts_exactly_up_to_its_bound();
  the_meter_never_hands_out_more_than_one_step();
  the_text_buffer_never_splits_a_code_point();
  the_text_buffer_reads_one_byte_pieces();
  invalid_and_unfinished_bytes_become_replacement_characters();
  return 0;
}
