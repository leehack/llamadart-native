#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_barrier_internal.h"
#include "llama_dart_tts_internal.h"

#include <cassert>
#include <new>
#include <stdexcept>
#include <string>

// The helpers behind the exception barrier, called directly for the cases
// that no exported function can be made to reach.

namespace {

std::string recorded(const std::string &message) {
  llama_dart_error_begin();
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

int g_freed = 0;

void count_free(void *object) {
  assert(object != nullptr);
  ++g_freed;
}

// A creating function frees what it made when tracking it throws, and only
// then.
void test_track_or_free() {
  int object = 0;
  const auto fail = [] { throw std::bad_alloc(); };

  g_freed = 0;
  assert(llama_dart_track_or_free(&object, count_free, [] {}) == &object);
  assert(g_freed == 0);

  bool thrown = false;
  try {
    llama_dart_track_or_free(&object, count_free, fail);
  } catch (const std::bad_alloc &) {
    thrown = true;
  }
  assert(thrown && g_freed == 1);

  // A call that created nothing has nothing to free.
  thrown = false;
  try {
    llama_dart_track_or_free(static_cast<int *>(nullptr), count_free, fail);
  } catch (const std::bad_alloc &) {
    thrown = true;
  }
  assert(thrown && g_freed == 1);
}

// An exception fails the TTS task, so that the next step does not continue
// it. A task without a generator, sampler or sequence has nothing to release
// and needs no model.
void test_tts_barrier() {
  llama_dart_tts task;
  task.state = LLAMA_DART_TTS_STATE_GENERATING;
  const int8_t flag = 0;
  task.cancel_flag = &flag;
  assert(llama_dart_tts_barrier(&task, []() -> llama_dart_tts_status {
           throw std::runtime_error("generation failed");
         }) == LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR);
  assert(task.state == LLAMA_DART_TTS_STATE_FAILED);
  assert(task.error == "generation failed");
  assert(task.cancel_flag == nullptr);
  assert(llama_dart_error.set);

  // A status that the call returns itself leaves the task as the call left
  // it.
  task.state = LLAMA_DART_TTS_STATE_GENERATING;
  task.error.clear();
  assert(llama_dart_tts_barrier(&task, [] {
           return LLAMA_DART_TTS_STATUS_INVALID_STATE;
         }) == LLAMA_DART_TTS_STATUS_INVALID_STATE);
  assert(task.state == LLAMA_DART_TTS_STATE_GENERATING && task.error.empty());
  assert(!llama_dart_error.set);

  // Without a task there is nothing to fail.
  assert(llama_dart_tts_barrier(nullptr, []() -> llama_dart_tts_status {
           throw std::runtime_error("no task");
         }) == LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR);
  assert(std::string(llama_dart_error.message) == "no task");
}

} // namespace

int main() {
  test_message_encoding();
  test_barrier();
  test_track_or_free();
  test_tts_barrier();
  return 0;
}
