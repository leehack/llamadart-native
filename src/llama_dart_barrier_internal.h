#pragma once

#include <cstdio>
#include <exception>

// The calling thread's last caught exception. It has no destructor and needs
// no allocation, so recording std::bad_alloc cannot fail and nothing is left
// to destroy when the thread or the process ends.
struct llama_dart_error_state {
  bool set;
  char message[512];
};

static thread_local llama_dart_error_state llama_dart_error = {};

static inline void llama_dart_error_record(const char *message) noexcept {
  snprintf(llama_dart_error.message, sizeof(llama_dart_error.message), "%s",
           message != nullptr ? message : "unknown C++ exception");
  llama_dart_error.set = true;
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
// leave the last error alone unless they catch an exception themselves.
template <typename Call>
static void llama_dart_free_barrier(Call &&call) noexcept {
  llama_dart_catch(false, [&call] {
    call();
    return true;
  });
}
