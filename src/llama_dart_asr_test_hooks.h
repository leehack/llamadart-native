#pragma once

#if defined(LLAMADART_TESTING)

#include "llama_dart_wrapper.h"

#include <limits>
#include <string_view>

struct llama_dart_asr_test_hooks {
  void *data = nullptr;
  size_t max_buffer_samples = std::numeric_limits<size_t>::max();
  void (*before_track)(void *) = nullptr;
  void (*before_release)(llama_dart_asr_stream *, void *) = nullptr;
  void (*before_queue_insert)() = nullptr;
  std::string_view old_segment_tail;
  std::string_view new_segment_head;
};

struct llama_dart_asr_test_input {
  void *data;
  int32_t (*push)(void *, const float *, size_t);
  int32_t (*flush)(void *);
  int32_t (*step)(void *);
};

extern "C" {

// The calling thread borrows the hooks until they are cleared.
LLAMADART_API void
llama_dart_test_asr_set_hooks(const llama_dart_asr_test_hooks *hooks);
LLAMADART_API llama_dart_asr_stream *
llama_dart_test_asr_input_stream(const llama_dart_asr_test_input *input);
}

#endif
