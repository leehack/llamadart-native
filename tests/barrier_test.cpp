#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_barrier_internal.h"

#include <cassert>
#include <new>
#include <stdexcept>
#include <string>

// The helpers behind the exception barrier, called directly for the cases
// that no exported function can be made to reach.

namespace {

std::string recorded(const std::string &message) {
  llama_dart_error.set = false;
  llama_dart_error_record(message.c_str());
  assert(llama_dart_error.set);
  return llama_dart_error.message;
}

// The last error is valid UTF-8 whatever the exception's message holds.
void test_message_encoding() {
  assert(recorded("plain") == "plain");
  assert(recorded("") == "");
  llama_dart_error_record(nullptr);
  assert(std::string(llama_dart_error.message) == "unknown C++ exception");

  const std::string replacement = "\xEF\xBF\xBD";
  // Well-formed sequences of every length are kept.
  const std::string valid = "a\xC3\xA9\xE2\x96\x81\xF0\x9F\x98\x80z";
  assert(recorded(valid) == valid);
  // The piece of a token may end inside a character.
  assert(recorded("piece: \xE2\x96 (7)") ==
         "piece: " + replacement + replacement + " (7)");
  assert(recorded("\xE2\x96") == replacement + replacement);
  // Bytes that start no sequence, an overlong form, a surrogate and a code
  // point above U+10FFFF.
  assert(recorded("\xFF") == replacement);
  assert(recorded("\x80z") == replacement + "z");
  assert(recorded("\xC0\xAF") == replacement + replacement);
  assert(recorded("\xE0\x80\xAF") == replacement + replacement + replacement);
  assert(recorded("\xED\xA0\x80") == replacement + replacement + replacement);
  assert(recorded("\xF4\x90\x80\x80") ==
         replacement + replacement + replacement + replacement);

  // A message that does not fit is cut between characters, not inside one.
  const size_t capacity = sizeof(llama_dart_error.message) - 1;
  assert(recorded(std::string(capacity + 100, 'a')) ==
         std::string(capacity, 'a'));
  assert(recorded(std::string(capacity - 1, 'a') + "\xC3\xA9") ==
         std::string(capacity - 1, 'a'));
  assert(recorded(std::string(capacity - 2, 'a') + "\xC3\xA9") ==
         std::string(capacity - 2, 'a') + "\xC3\xA9");
  assert(recorded(std::string(capacity - 2, 'a') + "\xFF") ==
         std::string(capacity - 2, 'a'));
  assert(recorded(std::string(capacity - 3, 'a') + "\xFF") ==
         std::string(capacity - 3, 'a') + replacement);
}

void test_barrier() {
  llama_dart_error_record("stale");
  assert(llama_dart_barrier(-1, [] { return 7; }) == 7);
  assert(!llama_dart_error.set);

  assert(llama_dart_barrier(-1, []() -> int {
           throw std::runtime_error("failed");
         }) == -1);
  assert(llama_dart_error.set);
  assert(std::string(llama_dart_error.message) == "failed");

  assert(llama_dart_barrier(-1, []() -> int { throw std::bad_alloc(); }) ==
         -1);
  assert(llama_dart_error.set && llama_dart_error.message[0] != '\0');

  assert(llama_dart_barrier(-1, []() -> int { throw 42; }) == -1);
  assert(std::string(llama_dart_error.message) == "unknown C++ exception");

  // A function that frees keeps an error it did not cause, and sets its own.
  llama_dart_free_barrier([] {});
  assert(std::string(llama_dart_error.message) == "unknown C++ exception");
  llama_dart_free_barrier([] { throw std::runtime_error("free failed"); });
  assert(std::string(llama_dart_error.message) == "free failed");
  llama_dart_void_barrier([] {});
  assert(!llama_dart_error.set);
}

} // namespace

int main() {
  test_message_encoding();
  test_barrier();
  return 0;
}
