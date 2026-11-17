#pragma once

#include "llama_dart_wrapper.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Pure parts of the live ASR session. They hold no llama.cpp state, so the
// tests in tests/asr_stream_meter_test.cpp run without a model.

static constexpr int32_t llama_dart_asr_stream_sample_rate = 16000;
static constexpr int32_t llama_dart_asr_stream_default_buffered_ms = 30000;

// Mirrors upstream's Voxtral Realtime stream: one text token per 80 ms row,
// after a prefix of BOS and 32 left-pad rows plus one row per delay token.
static constexpr int32_t llama_dart_asr_token_delay_row_ms = 80;
static constexpr int32_t llama_dart_asr_token_delay_left_pad_rows = 32;
// The audio one text token of LFM3-Audio live ASR covers: a chunk of 12 rows
// of 80 ms, and the longest batch is the prompt, a chunk and the mixed-end
// token.
static constexpr int32_t llama_dart_asr_chunk_ms = 960;
static constexpr int32_t llama_dart_asr_chunk_rows = 12;

static inline uint32_t
llama_dart_asr_stream_mode_bit(llama_dart_asr_stream_mode mode) {
  return 1u << static_cast<uint32_t>(mode);
}

// The two tests upstream's CLI and server route on: a stream with no chat is a
// token-delay model, a stream with an "asr" chat template is prompted. A
// stream that has neither only prefills a chat turn and writes no text.
static inline llama_dart_asr_stream_mode
llama_dart_asr_stream_classify(bool audio_stream, bool chat,
                               bool has_asr_template) {
  if (!audio_stream) {
    return LLAMA_DART_ASR_STREAM_MODE_NONE;
  }
  if (!chat) {
    return LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY;
  }
  return has_asr_template ? LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS
                          : LLAMA_DART_ASR_STREAM_MODE_NONE;
}

// NONE when the caller did not accept the mode.
static inline llama_dart_asr_stream_mode
llama_dart_asr_stream_accepted_mode(llama_dart_asr_stream_mode mode,
                                    uint32_t accepted_modes) {
  return mode != LLAMA_DART_ASR_STREAM_MODE_NONE &&
                 (accepted_modes & llama_dart_asr_stream_mode_bit(mode)) != 0
             ? mode
             : LLAMA_DART_ASR_STREAM_MODE_NONE;
}

// Rows the first decode of a stream puts in one batch, which n_batch must hold.
static inline int64_t
llama_dart_asr_stream_required_batch(llama_dart_asr_stream_mode mode,
                                     int64_t prompt_tokens,
                                     int64_t delay_tokens) {
  if (mode == LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY) {
    return 1 + llama_dart_asr_token_delay_left_pad_rows + delay_tokens;
  }
  return prompt_tokens + llama_dart_asr_chunk_rows + 1;
}

// Audio a token-delay session can hold before the context is full.
static inline int64_t
llama_dart_asr_stream_max_session_ms(int64_t context_tokens,
                                     int64_t prefix_rows) {
  return context_tokens > prefix_rows
             ? (context_tokens - prefix_rows) *
                   llama_dart_asr_token_delay_row_ms
             : 0;
}

// Samples of audio per upstream step. A token-delay stream encodes what it is
// given in one call, so the session gives it one step at a time: upstream
// itself encodes the whole backlog at once, which took 2.7 s for 297 s.
static inline size_t
llama_dart_asr_stream_step_samples(llama_dart_asr_stream_mode mode) {
  const int32_t ms = mode == LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS
                         ? llama_dart_asr_chunk_ms
                         : 4 * llama_dart_asr_token_delay_row_ms;
  return static_cast<size_t>(llama_dart_asr_stream_sample_rate) *
         static_cast<size_t>(ms) / 1000;
}

// A bounded queue between push and the upstream stream.
class llama_dart_asr_input_meter {
public:
  llama_dart_asr_input_meter(size_t capacity_samples, size_t step_samples)
      : capacity_(capacity_samples), step_(step_samples) {}

  // Takes as many of the samples as fit and returns how many.
  size_t accept(const float *samples, size_t count) {
    const size_t room = capacity_ - buffered();
    const size_t taken = std::min(room, count);
    if (taken > 0) {
      samples_.insert(samples_.end(), samples, samples + taken);
    }
    return taken;
  }

  size_t buffered() const { return samples_.size() - head_; }
  size_t capacity() const { return capacity_; }

  // The samples the next upstream step may be given, at most one step of them.
  size_t next_chunk(const float **out) const {
    *out = samples_.data() + head_;
    return std::min(step_, buffered());
  }

  void consume(size_t count) {
    head_ += std::min(count, buffered());
    if (head_ == samples_.size()) {
      samples_.clear();
      head_ = 0;
    } else if (head_ >= capacity_ / 2 + 1) {
      samples_.erase(samples_.begin(),
                     samples_.begin() + static_cast<std::ptrdiff_t>(head_));
      head_ = 0;
    }
  }

  void clear() {
    samples_.clear();
    head_ = 0;
  }

private:
  size_t capacity_;
  size_t step_;
  std::vector<float> samples_;
  size_t head_ = 0;
};

// Collects the pieces of sampled tokens and hands out whole code points. A
// token can end inside a character, so the tail of a piece waits for the next
// one. A byte that cannot start or continue a sequence is read as U+FFFD, so
// what is read is always valid UTF-8.
class llama_dart_asr_text_buffer {
public:
  void append(const char *bytes, size_t count) {
    pending_.append(bytes, count);
    drain(false);
  }

  // The stream has ended: a sequence that was left incomplete is read as
  // U+FFFD.
  void finish() { drain(true); }

  size_t available() const { return ready_.size() - head_; }

  // Moves up to capacity bytes to out, without splitting a code point, and
  // returns how many it moved.
  size_t read(char *out, size_t capacity) {
    size_t count = std::min(capacity, available());
    while (count > 0 && count < available() &&
           (static_cast<unsigned char>(ready_[head_ + count]) & 0xC0) == 0x80) {
      --count;
    }
    if (count > 0) {
      std::memcpy(out, ready_.data() + head_, count);
      head_ += count;
    }
    if (head_ == ready_.size()) {
      ready_.clear();
      head_ = 0;
    }
    return count;
  }

  void clear() {
    pending_.clear();
    ready_.clear();
    head_ = 0;
  }

private:
  // The length of the sequence pending_ starts at offset: its length when it
  // is complete, 0 when no sequence can start there, and -1 when it is the
  // prefix of one that more bytes could complete.
  int sequence_length(size_t offset) const {
    const auto byte = [&](size_t i) {
      return static_cast<unsigned char>(pending_[offset + i]);
    };
    const size_t have = pending_.size() - offset;
    const unsigned char lead = byte(0);
    if (lead < 0x80) {
      return 1;
    }
    size_t length = 0;
    unsigned char low = 0x80;
    unsigned char high = 0xBF;
    if (lead >= 0xC2 && lead <= 0xDF) {
      length = 2;
    } else if (lead >= 0xE0 && lead <= 0xEF) {
      length = 3;
      low = lead == 0xE0 ? 0xA0 : 0x80;
      high = lead == 0xED ? 0x9F : 0xBF;
    } else if (lead >= 0xF0 && lead <= 0xF4) {
      length = 4;
      low = lead == 0xF0 ? 0x90 : 0x80;
      high = lead == 0xF4 ? 0x8F : 0xBF;
    } else {
      return 0;
    }
    for (size_t i = 1; i < length; ++i) {
      if (i >= have) {
        return -1;
      }
      const unsigned char next = byte(i);
      const unsigned char min = i == 1 ? low : 0x80;
      const unsigned char max = i == 1 ? high : 0xBF;
      if (next < min || next > max) {
        return 0;
      }
    }
    return static_cast<int>(length);
  }

  void drain(bool final) {
    size_t offset = 0;
    while (offset < pending_.size()) {
      const int length = sequence_length(offset);
      if (length > 0) {
        ready_.append(pending_, offset, static_cast<size_t>(length));
        offset += static_cast<size_t>(length);
      } else if (length == 0) {
        ready_.append("\xEF\xBF\xBD");
        ++offset;
      } else if (final) {
        ready_.append("\xEF\xBF\xBD");
        offset = pending_.size();
      } else {
        break;
      }
    }
    pending_.erase(0, offset);
  }

  std::string pending_;
  std::string ready_;
  size_t head_ = 0;
};
