#pragma once

#include "llama_dart_wrapper.h"

#include <cstddef>
#include <cstdint>

namespace llama_dart_internal {
enum class session_kind { tts, speculative, mtp, ngram };

// Live allocation identities distinguish a valid untracked legacy handle from
// a late free of an owner-claimed handle, without retaining pointer tombstones.
LLAMADART_API uint64_t session_born(void *object, session_kind kind);
LLAMADART_API void session_died(void *object, uint64_t identity) noexcept;
LLAMADART_API size_t live_session_count();

struct session_lifetime {
  void *object;
  uint64_t identity;
  session_lifetime(void *object, session_kind kind)
      : object(object), identity(session_born(object, kind)) {}
  ~session_lifetime() { session_died(object, identity); }
  session_lifetime(const session_lifetime &) = delete;
  session_lifetime &operator=(const session_lifetime &) = delete;
};
} // namespace llama_dart_internal
