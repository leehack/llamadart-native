#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_asr_stream_internal.h"
#include "llama_dart_asr_test_hooks.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
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

void utf8_incomplete_prefix_is_discarded_at_segment_boundary() {
  llama_dart_asr_text_buffer text;
  // Complete word followed by an incomplete 3-byte UTF-8 prefix (U+20AC).
  text.append("hello\xE2\x82", 7);
  assert(text.available() == 5);

  // Segment break: discard incomplete prefix, preserve ready bytes.
  text.discard_incomplete();
  assert(text.available() == 5);

  // New segment starts with a continuation byte, then ASCII. It must not
  // complete U+20AC across the boundary; "hello" must survive.
  text.append("\xACworld", 6);
  char out[32];
  const size_t n = text.read(out, sizeof(out));
  const std::string result(out, n);
  assert(result == "hello\xEF\xBF\xBDworld");
  assert(result.find("\xE2\x82\xAC") == std::string::npos);
}

void capacity_conversion_rejects_overflow_and_nonpositive() {
  const size_t max_32bit_samples =
      std::numeric_limits<uint32_t>::max() / sizeof(float);
  size_t out = 0;

  // Default 30000 ms.
  assert(llama_dart_asr_stream_buffer_capacity(
      llama_dart_asr_stream_default_buffered_ms, max_32bit_samples, &out));
  assert(out == 480000);

  // Nonpositive values rejected.
  assert(!llama_dart_asr_stream_buffer_capacity(0, max_32bit_samples, &out));
  assert(!llama_dart_asr_stream_buffer_capacity(-1, max_32bit_samples, &out));
  assert(!llama_dart_asr_stream_buffer_capacity(
      std::numeric_limits<int32_t>::min(), max_32bit_samples, &out));

  // 268436 ms overflows 16000 * ms in uint32_t, but is 4,294,976 samples.
  assert(llama_dart_asr_stream_buffer_capacity(268436, max_32bit_samples,
                                               &out));
  assert(out == 4294976);

  // Representable 32-bit boundary and one above.
  const int32_t max_32bit_ms = static_cast<int32_t>(max_32bit_samples / 16);
  const size_t exact_capacity = static_cast<size_t>(max_32bit_ms) * 16;
  assert(llama_dart_asr_stream_buffer_capacity(max_32bit_ms, exact_capacity,
                                               &out));
  assert(out == exact_capacity);
  assert(!llama_dart_asr_stream_buffer_capacity(max_32bit_ms + 1,
                                                max_32bit_samples, &out));
}

void reset_clears_prior_cancel_before_release_and_preserves_later_cancel() {
  // Prior cancel is cleared and reset makes session reusable.
  std::atomic<bool> cancel_requested{true};
  bool released = false;
  bool state_reset = false;
  const bool ok = llama_dart_asr_reset_session(
      cancel_requested, [&] { released = true; }, [&] { state_reset = true; },
      [] { return true; });
  assert(ok && released && state_reset);
  assert(!cancel_requested.load(std::memory_order_acquire));

  // Cancel arriving after reset's entry boundary (while release is paused)
  // survives until the next step.
  std::atomic<bool> in_release{false};
  std::atomic<bool> allow_finish{false};
  std::thread canceller([&] {
    while (!in_release.load(std::memory_order_acquire)) {
    }
    cancel_requested.store(true, std::memory_order_release);
    allow_finish.store(true, std::memory_order_release);
  });
  const bool ok2 = llama_dart_asr_reset_session(
      cancel_requested,
      [&] {
        in_release.store(true, std::memory_order_release);
        while (!allow_finish.load(std::memory_order_acquire)) {
        }
      },
      [] {}, [] { return true; });
  canceller.join();
  assert(ok2);
  assert(cancel_requested.load(std::memory_order_acquire));
}

void step_feed_and_encode_pushes_at_most_one_chunk_and_steps_once() {
  const size_t step_samples =
      llama_dart_asr_stream_step_samples(LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS);
  llama_dart_asr_input_meter meter(step_samples * 5, step_samples);
  const std::vector<float> audio(step_samples * 3 + 4000, 0.25f);
  assert(meter.accept(audio.data(), audio.size()) == audio.size());

  llama_dart_asr_step_state state{};
  int push_calls = 0;
  int flush_calls = 0;
  int step_calls = 0;
  size_t last_pushed = 0;
  const int32_t scripted_rows[] = {0, 12, 1, 0, 0, 0, 5, 0, -1};
  size_t script_idx = 0;

  const auto push_fn = [&](const float *, size_t count) {
    ++push_calls;
    last_pushed = count;
    return true;
  };
  const auto flush_fn = [&]() {
    ++flush_calls;
    return true;
  };
  const auto step_fn = [&]() {
    ++step_calls;
    return scripted_rows[script_idx++];
  };

  int32_t rows = 0;

  // Step 1: pushes 1 chunk, step returns 0 (needs_audio).
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 1 && last_pushed == step_samples && step_calls == 1);
  assert(meter.buffered() == step_samples * 2 + 4000);
  assert(state.needs_audio);

  // Step 2: pushes 1 chunk, step returns 12 (rows_ready).
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::rows_ready);
  assert(rows == 12 && push_calls == 2 && last_pushed == step_samples &&
         step_calls == 2);
  assert(meter.buffered() == step_samples + 4000);
  assert(!state.needs_audio);

  // Step 3: needs_audio is false -> 0 pushes, 1 step returning 1.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::rows_ready);
  assert(rows == 1 && push_calls == 2 && step_calls == 3);
  assert(meter.buffered() == step_samples + 4000);

  // Step 4: needs_audio is false -> 0 pushes, 1 step returning 0 (needs_audio).
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 2 && step_calls == 4 && state.needs_audio);

  // Step 5: pushes 1 chunk, step returns 0.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 3 && last_pushed == step_samples && step_calls == 5);
  assert(meter.buffered() == 4000);

  // Step 6: pushes remaining 4000 samples, step returns 0.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 4 && last_pushed == 4000 && step_calls == 6);
  assert(meter.buffered() == 0);

  // Step 7: empty queue & not finished -> 0 pushes, 0 flushes, 0 steps.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 4 && flush_calls == 0 && step_calls == 6);

  // Step 8: input_finished -> flushes once, steps once returning 5.
  state.input_finished = true;
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::rows_ready);
  assert(rows == 5 && push_calls == 4 && flush_calls == 1 && step_calls == 7);

  // Upstream can ask for more audio after the final flush.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::needs_audio);
  assert(push_calls == 4 && flush_calls == 1 && step_calls == 8);
  assert(state.needs_audio);

  // Step 10: even after a zero-row step, never flush twice.
  assert(llama_dart_asr_step_feed_and_encode(meter, state, push_fn, flush_fn,
                                             step_fn, &rows) ==
         llama_dart_asr_feed_outcome::finished);
  assert(push_calls == 4 && flush_calls == 1 && step_calls == 9);
}

void public_step_pushes_only_one_chunk_of_backlog() {
  struct calls {
    std::vector<size_t> pushes;
    int flushes = 0;
    int steps = 0;
  } observed;
  const llama_dart_asr_test_input input{
      &observed,
      [](void *data, const float *, size_t count) {
        static_cast<calls *>(data)->pushes.push_back(count);
        return 0;
      },
      [](void *data) {
        ++static_cast<calls *>(data)->flushes;
        return 0;
      },
      [](void *data) {
        auto &state = *static_cast<calls *>(data);
        return ++state.steps == 5 ? -1 : 0;
      },
  };
  auto *stream = llama_dart_test_asr_input_stream(&input);
  const size_t chunk = kRate * 960 / 1000;
  const size_t tail = kRate / 10;
  const std::vector<float> backlog(2 * chunk + tail, 0.25f);
  size_t accepted = 0;
  assert(llama_dart_asr_stream_push(stream, backlog.data(), backlog.size(),
                                    &accepted) ==
         LLAMA_DART_ASR_STREAM_STATUS_OK);
  assert(accepted == backlog.size());
  assert(llama_dart_asr_stream_finish(stream) ==
         LLAMA_DART_ASR_STREAM_STATUS_OK);

  llama_dart_asr_stream_progress progress{};
  progress.struct_size = sizeof(progress);
  const size_t expected_pushes[] = {chunk, chunk, tail};
  int64_t consumed = 0;
  for (size_t count : expected_pushes) {
    const size_t previous_pushes = observed.pushes.size();
    assert(llama_dart_asr_stream_step(stream, &progress) ==
           LLAMA_DART_ASR_STREAM_STATUS_OK);
    consumed += static_cast<int64_t>(count);
    assert(observed.pushes.size() == previous_pushes + 1);
    assert(observed.pushes.back() == count && observed.flushes == 0);
    assert(observed.steps == static_cast<int>(observed.pushes.size()));
    assert(progress.samples_consumed == consumed);
    assert(progress.samples_buffered ==
           static_cast<int64_t>(backlog.size()) - consumed);
    assert(progress.state == LLAMA_DART_ASR_STREAM_STATE_WORKING);
  }
  assert(llama_dart_asr_stream_step(stream, &progress) ==
         LLAMA_DART_ASR_STREAM_STATUS_OK);
  assert(observed.pushes.size() == 3 && observed.flushes == 1 &&
         observed.steps == 4);
  assert(progress.state == LLAMA_DART_ASR_STREAM_STATE_WORKING);
  assert(llama_dart_asr_stream_step(stream, &progress) ==
         LLAMA_DART_ASR_STREAM_STATUS_OK);
  assert(observed.pushes.size() == 3 && observed.flushes == 1 &&
         observed.steps == 5);
  assert(progress.state == LLAMA_DART_ASR_STREAM_STATE_FINISHED);
  llama_dart_asr_stream_free(stream);
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
  utf8_incomplete_prefix_is_discarded_at_segment_boundary();
  capacity_conversion_rejects_overflow_and_nonpositive();
  reset_clears_prior_cancel_before_release_and_preserves_later_cancel();
  step_feed_and_encode_pushes_at_most_one_chunk_and_steps_once();
  public_step_pushes_only_one_chunk_of_backlog();
  return 0;
}
