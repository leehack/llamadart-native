#pragma once

#include <cstddef>
#include <cstring>
#include <exception>

// The calling thread's last caught exception. It has no destructor, so nothing
// is left to destroy when the thread or the process ends, and recording into
// it allocates nothing. The platform may allocate the storage itself on the
// thread's first access: emulated thread-local storage on Android does, and so
// does the dynamic loader for a library loaded with dlopen. A barrier that
// clears the last error reaches the storage before the call that may throw,
// so that recording std::bad_alloc afterwards finds it there. A barrier of a
// function that frees does not: see llama_dart_free_barrier. If the platform
// cannot allocate the storage, it aborts the process.
struct llama_dart_error_state {
  bool set;
  char message[512];
};

static thread_local llama_dart_error_state llama_dart_error = {};

// Number of bytes of the well-formed UTF-8 sequence that text starts with, or
// 0 when it starts with none. It reads no byte after a terminating zero.
static inline size_t llama_dart_utf8_sequence_length(const unsigned char *text) {
  const unsigned char lead = text[0];
  if (lead < 0x80) {
    return lead != 0 ? 1 : 0;
  }
  size_t length = 0;
  unsigned char second_low = 0x80;
  unsigned char second_high = 0xBF;
  if (lead >= 0xC2 && lead <= 0xDF) {
    length = 2;
  } else if (lead >= 0xE0 && lead <= 0xEF) {
    length = 3;
    second_low = lead == 0xE0 ? 0xA0 : 0x80;
    second_high = lead == 0xED ? 0x9F : 0xBF;
  } else if (lead >= 0xF0 && lead <= 0xF4) {
    length = 4;
    second_low = lead == 0xF0 ? 0x90 : 0x80;
    second_high = lead == 0xF4 ? 0x8F : 0xBF;
  } else {
    return 0;
  }
  if (text[1] < second_low || text[1] > second_high) {
    return 0;
  }
  for (size_t i = 2; i < length; ++i) {
    if ((text[i] & 0xC0) != 0x80) {
      return 0;
    }
  }
  return length;
}

// Records message as valid UTF-8, so that a caller can decode it: a message
// from llama.cpp may hold part of a character, as the piece of a token does.
// Each byte that starts no well-formed sequence becomes U+FFFD, and a message
// that does not fit is cut between characters.
#if defined(_MSC_VER)
#define LLAMA_DART_NOINLINE __declspec(noinline)
#else
#define LLAMA_DART_NOINLINE __attribute__((noinline))
#endif

static LLAMA_DART_NOINLINE void
llama_dart_error_record(const char *message) noexcept {
  llama_dart_error_state &error = llama_dart_error;
  const auto *source = reinterpret_cast<const unsigned char *>(
      message != nullptr ? message : "unknown C++ exception");
  const size_t capacity = sizeof(error.message) - 1;
  size_t used = 0;
  while (*source != 0) {
    const size_t sequence = llama_dart_utf8_sequence_length(source);
    const char *bytes = sequence != 0 ? reinterpret_cast<const char *>(source)
                                      : "\xEF\xBF\xBD";
    const size_t length = sequence != 0 ? sequence : 3;
    if (used + length > capacity) {
      break;
    }
    memcpy(error.message + used, bytes, length);
    used += length;
    source += sequence != 0 ? sequence : 1;
  }
  error.message[used] = '\0';
  error.set = true;
}

// Runs call so that no C++ exception leaves libllamadart: an exception unwinds
// call, is recorded as the thread's last error, and failure is returned.
template <typename Result, typename Call>
static Result llama_dart_catch(Result failure, Call &&call) noexcept {
  try {
    return call();
  } catch (const std::exception &error) {
    llama_dart_error_record(error.what());
  } catch (...) {
    llama_dart_error_record(nullptr);
  }
  return failure;
}

// The same for a call whose caller reads the last error afterwards: it starts
// with none, so that a last error after the call is this call's.
template <typename Result, typename Call>
static Result llama_dart_barrier(Result failure, Call &&call) noexcept {
  llama_dart_error.set = false;
  return llama_dart_catch(failure, call);
}

template <typename Call>
static void llama_dart_void_barrier(Call &&call) noexcept {
  llama_dart_barrier(false, [&call] {
    call();
    return true;
  });
}

// For the functions that free an object. A Dart finalizer may run one on a
// thread between a call that failed there and the read of its error, so they
// leave the last error alone unless they catch an exception themselves. They
// therefore first reach the thread's storage when they record an exception.
// Reading it before the call, to have it allocated by then, made the
// Windows ARM64 build crash when a free function threw.
template <typename Call>
static void llama_dart_free_barrier(Call &&call) noexcept {
  llama_dart_catch(false, [&call] {
    call();
    return true;
  });
}

// Runs track for an object that a creating call just made and returns the
// object. When track throws, the object is freed first, so that an exception
// leaves nothing behind that exit teardown does not know.
template <typename Object, typename Track>
static Object *llama_dart_track_or_free(Object *object, void (*free_fn)(void *),
                                        Track &&track) {
  try {
    track();
  } catch (...) {
    if (object != nullptr) {
      free_fn(object);
    }
    throw;
  }
  return object;
}
