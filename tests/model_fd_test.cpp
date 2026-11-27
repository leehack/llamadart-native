#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_wrapper.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

constexpr int32_t kSuccess = 0;
constexpr int32_t kInvalidArgument = 1;
constexpr int32_t kUnsupported = 2;
constexpr int32_t kInvalidDescriptor = 3;
constexpr int32_t kSizeMismatch = 4;
constexpr int32_t kLoadFailed = 5;
constexpr int32_t kError = 6;

llama_model_params cpu_params(llama_load_mode mode = LLAMA_LOAD_MODE_MMAP) {
  static ggml_backend_dev_t no_devices[] = {nullptr};
  llama_model_params params = llama_model_default_params();
  params.n_gpu_layers = 0;
  params.devices = no_devices;
  params.load_mode = mode;
  return params;
}

bool last_error_contains(const char *text) {
  const char *error = llama_dart_last_error();
  return error != nullptr && std::strstr(error, text) != nullptr;
}

#if defined(_WIN32)

int test_unsupported() {
  assert(llama_dart_model_fd_api_version() == 0);
  llama_model *model = nullptr;
  assert(llama_dart_model_load_from_fd(3, -1, cpu_params(), &model) ==
         kUnsupported);
  assert(model == nullptr && last_error_contains("not supported"));
  assert(llama_dart_model_load_from_fd(3, -1, cpu_params(), nullptr) ==
         kInvalidArgument);
  assert(llama_dart_exit_tracked_count() == 0);
  return 0;
}

}  // namespace

int main() { return test_unsupported(); }

#else

void init_backends() {
#if defined(LLAMADART_TEST_CPU_BACKEND)
  assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
  llama_backend_init();
}

std::string temp_path(const char *name) {
  const char *directory = std::getenv("TMPDIR");
  return std::string(directory != nullptr ? directory : "/tmp") + "/llamadart_" +
         name + "_" + std::to_string(getpid());
}

bool is_open(int fd) { return fcntl(fd, F_GETFD) != -1; }

off_t offset_of(int fd) { return lseek(fd, 0, SEEK_CUR); }

int open_read_only(const char *path) {
  const int fd = open(path, O_RDONLY | O_CLOEXEC);
  assert(fd >= 0);
  return fd;
}

struct scratch_file {
  std::string path;
  int fd = -1;
  explicit scratch_file(const char *name, const std::string &content,
                        int flags = O_RDWR) {
    path = temp_path(name);
    const int writer = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    assert(writer >= 0);
    assert(write(writer, content.data(), content.size()) ==
           static_cast<ssize_t>(content.size()));
    close(writer);
    fd = open(path.c_str(), flags | O_CLOEXEC);
    assert(fd >= 0);
  }
  ~scratch_file() {
    if (fd >= 0) close(fd);
    unlink(path.c_str());
  }
};

// Failures must name their cause, leave out_model NULL, track nothing and leave
// the caller's descriptor open and where it was.
void expect_refused(int fd, int64_t size, int32_t status, const char *cause) {
  llama_model *model = reinterpret_cast<llama_model *>(0x1);
  const off_t before = fd >= 0 && is_open(fd) ? offset_of(fd) : -1;
  assert(llama_dart_model_load_from_fd(fd, size, cpu_params(), &model) ==
         status);
  assert(model == nullptr);
  assert(last_error_contains(cause));
  assert(llama_dart_exit_tracked_count() == 0);
  if (fd >= 0 && before != -1) {
    assert(is_open(fd) && offset_of(fd) == before);
  }
}

int test_contract() {
  init_backends();
  assert(llama_dart_model_fd_api_version() == 1);
  llama_model *model = nullptr;
  assert(llama_dart_model_load_from_fd(0, -1, cpu_params(), nullptr) ==
         kInvalidArgument);
  assert(last_error_contains("out_model"));

  expect_refused(-1, -1, kInvalidDescriptor, "descriptor");
  scratch_file closed("closed", "x");
  const int stale = dup(closed.fd);
  close(stale);
  expect_refused(stale, -1, kInvalidDescriptor, "open");

  const int directory = open("/", O_RDONLY);
  assert(directory >= 0);
  expect_refused(directory, -1, kInvalidDescriptor, "regular");
  close(directory);

  int pipe_ends[2];
  assert(pipe(pipe_ends) == 0);
  expect_refused(pipe_ends[0], -1, kInvalidDescriptor, "regular");
  close(pipe_ends[0]);
  close(pipe_ends[1]);

  scratch_file write_only("writeonly", "GGUF", O_WRONLY);
  expect_refused(write_only.fd, -1, kInvalidDescriptor, "readable");

  const std::string content(4096, 'x');
  scratch_file not_gguf("notgguf", content);
  expect_refused(not_gguf.fd, -1, kLoadFailed, "GGUF");
  expect_refused(not_gguf.fd, static_cast<int64_t>(content.size()), kLoadFailed,
                 "GGUF");
  expect_refused(not_gguf.fd, static_cast<int64_t>(content.size()) + 1,
                 kSizeMismatch, "size");
  expect_refused(not_gguf.fd, 0, kSizeMismatch, "size");

  scratch_file empty("empty", "");
  expect_refused(empty.fd, -1, kLoadFailed, "GGUF");

  assert(llama_dart_model_load_from_fd(not_gguf.fd, -1, cpu_params(), &model) ==
         kLoadFailed);
  llama_dart_clear_last_error();
  assert(llama_dart_last_error() == nullptr);
  void *pending = llama_dart_owner_create();
  assert(llama_dart_owner_model_load_from_fd(pending, not_gguf.fd, -1,
                                             cpu_params(), &model) == kError);
  assert(model == nullptr && last_error_contains("not active"));
  llama_dart_owner_abandon(pending);
  return 0;
}

std::vector<float> first_logits(llama_model *model) {
  llama_context_params params = llama_context_default_params();
  params.n_ctx = 256;
  params.n_threads = params.n_threads_batch = 2;
  llama_context *context = llama_dart_init_from_model(model, params);
  assert(context != nullptr);
  std::vector<llama_token> tokens;
  for (llama_token token = 1; token <= 8; ++token) tokens.push_back(token);
  assert(llama_dart_decode(context, llama_batch_get_one(
                                        tokens.data(),
                                        static_cast<int32_t>(tokens.size()))) ==
         0);
  const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  const float *logits = llama_get_logits_ith(context, -1);
  assert(logits != nullptr && n_vocab > 0);
  std::vector<float> copy(logits, logits + n_vocab);
  llama_dart_exit_free(context);
  return copy;
}

void expect_same_as_path_load(const char *path, llama_model *from_fd,
                              llama_load_mode mode) {
  llama_model *from_path = llama_dart_model_load_from_file(path, cpu_params(mode));
  assert(from_path != nullptr);
  assert(first_logits(from_fd) == first_logits(from_path));
  llama_dart_exit_free(from_path);
}

bool refuse_load(float, void *) { return false; }

struct load_gate {
  std::mutex mutex;
  std::condition_variable changed;
  bool inside = false;
  bool proceed = false;
};

bool hold_load(float, void *data) {
  auto &gate = *static_cast<load_gate *>(data);
  std::unique_lock<std::mutex> lock(gate.mutex);
  gate.inside = true;
  gate.changed.notify_all();
  gate.changed.wait(lock, [&] { return gate.proceed; });
  return true;
}

// A descriptor load is a call in flight: teardown must wait for it, and the
// load must end instead of finishing after teardown freed its neighbours.
int test_teardown_waits(const char *path) {
  init_backends();
  load_gate gate;
  llama_model_params params = cpu_params();
  params.progress_callback = hold_load;
  params.progress_callback_user_data = &gate;
  const int fd = open_read_only(path);
  std::thread loader([&] {
    llama_model *model = nullptr;
    llama_dart_model_load_from_fd(fd, -1, params, &model);
    // A call that was in flight at teardown never returns to its caller.
    std::abort();
  });
  loader.detach();
  {
    std::unique_lock<std::mutex> lock(gate.mutex);
    assert(gate.changed.wait_for(lock, std::chrono::seconds(10),
                                 [&] { return gate.inside; }));
  }
  llama_dart_exit_set_wait_ms(10000);
  std::atomic<bool> torn_down{false};
  std::thread teardown([&] {
    llama_dart_exit_teardown();
    torn_down = true;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  assert(!torn_down.load());
  {
    std::lock_guard<std::mutex> lock(gate.mutex);
    gate.proceed = true;
    gate.changed.notify_all();
  }
  teardown.join();
  assert(torn_down.load() && llama_dart_exit_tracked_count() == 0);
  std::_Exit(0);
}

int test_model(const char *path) {
  init_backends();
  struct stat info;
  assert(stat(path, &info) == 0);
  const int64_t size = info.st_size;

  for (const llama_load_mode mode : {LLAMA_LOAD_MODE_MMAP, LLAMA_LOAD_MODE_NONE}) {
    const int fd = open_read_only(path);
    // Loads from offset 0 whatever the caller's offset, and restores it.
    assert(lseek(fd, 100, SEEK_SET) == 100);
    llama_model *model = nullptr;
    assert(llama_dart_model_load_from_fd(fd, size, cpu_params(mode), &model) ==
           kSuccess);
    assert(model != nullptr && llama_dart_exit_tracked_count() == 1);
    // The caller keeps its descriptor: still open, same offset.
    assert(is_open(fd) && offset_of(fd) == 100);

    // The model needs no descriptor after the call: a mapped model stays
    // valid with the caller's descriptor closed.
    close(fd);
    expect_same_as_path_load(path, model, mode);
    llama_dart_exit_free(model);
    assert(llama_dart_exit_tracked_count() == 0);
  }

  {
    // The caller's progress callback still runs and can cancel the load.
    const int cancel_fd = open_read_only(path);
    llama_model_params cancel_params = cpu_params();
    cancel_params.progress_callback = refuse_load;
    llama_model *cancelled = reinterpret_cast<llama_model *>(0x1);
    assert(llama_dart_model_load_from_fd(cancel_fd, size, cancel_params,
                                         &cancelled) == kLoadFailed);
    assert(cancelled == nullptr && llama_dart_exit_tracked_count() == 0);
    assert(is_open(cancel_fd) && offset_of(cancel_fd) == 0);
    close(cancel_fd);
  }

  const int fd = open_read_only(path);
  llama_model *first = nullptr;
  llama_model *second = nullptr;
  assert(llama_dart_model_load_from_fd(fd, -1, cpu_params(), &first) == kSuccess);
  assert(llama_dart_model_load_from_fd(fd, size, cpu_params(), &second) ==
         kSuccess);
  llama_model *refused = reinterpret_cast<llama_model *>(0x1);
  assert(llama_dart_model_load_from_fd(fd, size + 1, cpu_params(), &refused) ==
         kSizeMismatch);
  assert(refused == nullptr && is_open(fd));
  llama_dart_exit_free(first);
  assert(is_open(fd));
  close(fd);
  llama_dart_exit_free(second);
  assert(llama_dart_exit_tracked_count() == 0);

  void *owner = llama_dart_owner_create();
  assert(owner != nullptr && llama_dart_owner_activate(owner));
  const int owned_fd = open_read_only(path);
  llama_model *owned = nullptr;
  assert(llama_dart_owner_model_load_from_fd(owner, owned_fd, size,
                                             cpu_params(), &owned) == kSuccess);
  assert(owned != nullptr && llama_dart_exit_tracked_count() == 1);
  assert(is_open(owned_fd));
  llama_dart_owner_release(owner);
  assert(llama_dart_exit_tracked_count() == 0);
  assert(is_open(owned_fd));
  close(owned_fd);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "teardown") == 0) {
    return test_teardown_waits(argv[2]);
  }
  if (argc == 2) return test_model(argv[1]);
  assert(argc == 1);
  return test_contract();
}

#endif
