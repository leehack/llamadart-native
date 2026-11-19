#include "llama_dart_wrapper.h"

#if defined(LLAMADART_TESTING)
#include "llama_dart_asr_test_hooks.h"
#endif

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kModes = (1u << LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY) |
                            (1u << LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS);
constexpr const char *kExpected =
    "and so my fellow americans ask not what your country can do for you ask "
    "what you can do for your country";

struct options {
  const char *model = nullptr;
  const char *mmproj = nullptr;
  const char *audio = nullptr;
  bool gpu = false;
  int threads = 4;
  int context = 4096;
  double max_step_ms = 500.0;
  double long_seconds = 297.0;
};

struct run_result {
  std::string text;
  llama_dart_asr_stream_status status = LLAMA_DART_ASR_STREAM_STATUS_OK;
  llama_dart_asr_stream_progress progress{};
  double first_token_step_ms = 0.0;
  double steady_max_step_ms = 0.0;
  double max_step_ms = 0.0;
  double wall_s = 0.0;
  int steps = 0;
};

[[noreturn]] void fail(const char *what) {
  std::fprintf(stderr, "FAIL %s\n", what);
  std::exit(1);
}

void check(bool ok, const char *what) {
  if (!ok) {
    fail(what);
  }
}

void quiet_log(ggml_log_level level, const char *text, void *) {
  if (level >= GGML_LOG_LEVEL_ERROR) {
    std::fputs(text, stderr);
  }
}

double now_ms() {
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

// 0 where there is no getrusage, which makes the memory check pass.
double peak_rss_mb() {
#if defined(_WIN32)
  return 0.0;
#else
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#if defined(__APPLE__)
  return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#endif
}

std::string normalize(const std::string &text) {
  std::string out;
  bool space = false;
  for (unsigned char c : text) {
    if (std::isalnum(c)) {
      if (space && !out.empty()) {
        out.push_back(' ');
      }
      space = false;
      out.push_back(static_cast<char>(std::tolower(c)));
    } else {
      space = true;
    }
  }
  return out;
}

// chunk: 0 = one model step of samples, 1 = everything at once, else a random
// size in [1 ms, 1.5 s] from the seed.
struct chunker {
  size_t mode;
  std::mt19937 rng;
  size_t next(size_t rate, size_t remaining) {
    if (mode == 1) {
      return remaining;
    }
    if (mode == 0) {
      return std::min(remaining, rate * 320 / 1000);
    }
    return std::min(remaining, std::uniform_int_distribution<size_t>(
                                   rate / 1000, rate * 3 / 2)(rng));
  }
};

void drain_text(llama_dart_asr_stream *stream, int32_t *last_breaks,
                int32_t breaks, std::string *text) {
  size_t length = 0;
  check(llama_dart_asr_stream_read_text(stream, nullptr, 0, &length) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "text query");
  if (length == 0) {
    return;
  }
  std::string piece(length, '\0');
  size_t read = 0;
  check(llama_dart_asr_stream_read_text(stream, piece.data(), piece.size() + 4,
                                        &read) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "text read");
  piece.resize(read);
  if (breaks != *last_breaks && !text->empty()) {
    text->push_back(' ');
  }
  *last_breaks = breaks;
  text->append(piece);
}

// Pushes `pcm` in chunks, offers the rest again when the queue is full, and
// steps until the session ends.
run_result run(llama_dart_asr_stream *stream, const std::vector<float> &pcm,
               size_t chunk_mode, unsigned seed, size_t rate) {
  run_result result;
  chunker chunks{chunk_mode, std::mt19937(seed)};
  size_t offered = 0;
  size_t pending = 0;
  bool finished = false;
  int32_t last_breaks = 0;
  const double start = now_ms();
  for (;;) {
    if (offered < pcm.size()) {
      if (pending == 0) {
        pending = chunks.next(rate, pcm.size() - offered);
      }
      size_t accepted = 0;
      check(llama_dart_asr_stream_push(stream, pcm.data() + offered, pending,
                                       &accepted) ==
                LLAMA_DART_ASR_STREAM_STATUS_OK,
            "push");
      offered += accepted;
      pending -= accepted;
    } else if (!finished) {
      check(llama_dart_asr_stream_finish(stream) ==
                LLAMA_DART_ASR_STREAM_STATUS_OK,
            "finish");
      finished = true;
    }
    llama_dart_asr_stream_progress progress{};
    progress.struct_size = sizeof(progress);
    const double before = now_ms();
    const int32_t prev_tokens = result.progress.tokens_sampled;
    result.status = llama_dart_asr_stream_step(stream, &progress);
    const double step_ms = now_ms() - before;
    result.max_step_ms = std::max(result.max_step_ms, step_ms);
    if (prev_tokens == 0 && progress.tokens_sampled > 0) {
      result.first_token_step_ms = step_ms;
    } else if (prev_tokens > 0) {
      result.steady_max_step_ms = std::max(result.steady_max_step_ms, step_ms);
    }
    ++result.steps;
    result.progress = progress;
    drain_text(stream, &last_breaks, progress.segment_breaks, &result.text);
    if (result.status != LLAMA_DART_ASR_STREAM_STATUS_OK ||
        progress.state == LLAMA_DART_ASR_STREAM_STATE_FINISHED) {
      break;
    }
  }
  result.wall_s = (now_ms() - start) / 1000.0;
  return result;
}

std::vector<float> repeated(const std::vector<float> &pcm, size_t samples) {
  std::vector<float> out;
  out.reserve(samples);
  while (out.size() < samples) {
    const size_t take = std::min(pcm.size(), samples - out.size());
    out.insert(out.end(), pcm.begin(), pcm.begin() + take);
  }
  return out;
}

struct stream_deleter {
  void operator()(llama_dart_asr_stream *stream) const {
    llama_dart_asr_stream_free(stream);
  }
};
using stream_ptr = std::unique_ptr<llama_dart_asr_stream, stream_deleter>;

stream_ptr open_stream(llama_context *context, mtmd_context *mtmd,
                       const llama_dart_asr_stream_params *params = nullptr) {
  llama_dart_asr_stream_status status = LLAMA_DART_ASR_STREAM_STATUS_OK;
  stream_ptr stream(
      llama_dart_asr_stream_init(context, mtmd, kModes, params, &status));
  check(stream != nullptr && status == LLAMA_DART_ASR_STREAM_STATUS_OK,
        "stream init");
  return stream;
}

} // namespace

int main(int argc, char **argv) {
  options opt;
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--gpu") {
      opt.gpu = true;
    } else if (arg == "--threads" && i + 1 < argc) {
      opt.threads = std::atoi(argv[++i]);
    } else if (arg == "--context" && i + 1 < argc) {
      opt.context = std::atoi(argv[++i]);
    } else if (arg == "--max-step-ms" && i + 1 < argc) {
      opt.max_step_ms = std::atof(argv[++i]);
    } else if (arg == "--long-seconds" && i + 1 < argc) {
      opt.long_seconds = std::atof(argv[++i]);
    } else if (positional == 0) {
      opt.model = argv[i], ++positional;
    } else if (positional == 1) {
      opt.mmproj = argv[i], ++positional;
    } else if (positional == 2) {
      opt.audio = argv[i], ++positional;
    }
  }
  if (positional != 3) {
    std::fprintf(stderr,
                 "usage: %s MODEL MMPROJ AUDIO.wav [--gpu] [--threads N] "
                 "[--context N] [--max-step-ms MS] [--long-seconds S]\n",
                 argv[0]);
    return 2;
  }

  llama_backend_init();
  llama_dart_set_log_level(4);
  mtmd_log_set(quiet_log, nullptr);
  mtmd_helper_log_set(quiet_log, nullptr);
  const int32_t tracked_start = llama_dart_exit_tracked_count();

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = opt.gpu ? 99 : 0;
  llama_model *model = llama_dart_model_load_from_file(opt.model, model_params);
  check(model != nullptr, "model load");
  const auto make_context = [&](int n_ctx, uint32_t sequences = 1) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = static_cast<uint32_t>(n_ctx);
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_seq_max = sequences;
    params.n_threads = opt.threads;
    params.n_threads_batch = opt.threads;
    llama_context *context = llama_dart_init_from_model(model, params);
    check(context != nullptr, "context init");
    return context;
  };
  llama_context *context = make_context(opt.context);
  const uint32_t context_tokens = llama_n_ctx_seq(context);
  check(context_tokens == llama_n_ctx(context),
        "the default context was split across sequences");
  mtmd_context_params mtmd_params = mtmd_context_params_default();
  mtmd_params.use_gpu = opt.gpu;
  mtmd_params.n_threads = opt.threads;
  mtmd_params.print_timings = false;
  mtmd_context *mtmd =
      llama_dart_mtmd_init_from_file(opt.mmproj, model, &mtmd_params);
  check(mtmd != nullptr, "mmproj load");
  const int32_t tracked_base = llama_dart_exit_tracked_count();

  llama_dart_asr_stream_info info{};
  info.struct_size = sizeof(info);
  check(llama_dart_asr_stream_get_info(context, mtmd, kModes, &info) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "get_info");
  check(info.sample_rate == 16000 && info.channels == 1, "info format");
  const bool token_delay = info.mode == LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY;
  check(info.mode == (token_delay ? LLAMA_DART_ASR_STREAM_MODE_TOKEN_DELAY
                                  : LLAMA_DART_ASR_STREAM_MODE_PROMPTED_CHUNKS),
        "unexpected stream mode");
  if (token_delay) {
    check(info.max_session_ms ==
              (static_cast<int64_t>(context_tokens) - 39) * 80,
          "token-delay duration did not match the sequence context");
  } else {
    check(info.max_session_ms == 0, "prompted stream has a fixed duration");
    check(info.delay_ms == 960 && info.text_interval_ms == 960,
          "prompted delay must report the nominal 960 ms cadence");
  }
  std::printf("mode=%s text_interval_ms=%d delay_ms=%d max_session_ms=%lld\n",
              token_delay ? "TOKEN_DELAY" : "PROMPTED_CHUNKS",
              info.text_interval_ms, info.delay_ms,
              static_cast<long long>(info.max_session_ms));
  check(llama_dart_asr_stream_get_info(context, mtmd, 0, &info) ==
            LLAMA_DART_ASR_STREAM_STATUS_UNSUPPORTED,
        "an empty mask accepted the model");

  std::ifstream file(opt.audio, std::ios::binary);
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
      llama_dart_mtmd_bitmap_init_from_buf(mtmd, bytes.data(), bytes.size()),
      mtmd_bitmap_free);
  check(bitmap != nullptr && mtmd_bitmap_is_audio(bitmap.get()), "audio decode");
  const auto *samples =
      reinterpret_cast<const float *>(mtmd_bitmap_get_data(bitmap.get()));
  const std::vector<float> clip(
      samples, samples + mtmd_bitmap_get_n_bytes(bitmap.get()) / sizeof(float));
  const size_t rate = static_cast<size_t>(info.sample_rate);

  // Sequence ownership: init claims and clears sequence 0 while leaving
  // unrelated sequence 1 intact.
  {
    llama_context *two_sequences = make_context(512, 2);
    llama_memory_t mem = llama_get_memory(two_sequences);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab);
    if (bos == LLAMA_TOKEN_NULL) {
      bos = 0;
    }
    for (llama_seq_id seq = 0; seq < 2; ++seq) {
      llama_batch seed = llama_batch_init(1, 0, 1);
      seed.n_tokens = 1;
      seed.token[0] = bos;
      seed.pos[0] = 0;
      seed.n_seq_id[0] = 1;
      seed.seq_id[0][0] = seq;
      seed.logits[0] = 0;
      check(llama_decode(two_sequences, seed) == 0, "seed sequence");
      llama_batch_free(seed);
      check(llama_memory_seq_pos_max(mem, seq) >= 0, "seeded sequence missing");
    }
    stream_ptr seq_check = open_stream(two_sequences, mtmd);
    check(llama_memory_seq_pos_max(mem, 0) < 0,
          "init did not clear chosen sequence 0");
    check(llama_memory_seq_pos_max(mem, 1) >= 0,
          "init cleared unrelated sequence 1");
    seq_check.reset();
    llama_dart_exit_free(two_sequences);
    std::printf("sequence ownership: seq 0 cleared, seq 1 preserved\n");
  }

  stream_ptr stream = open_stream(context, mtmd);
  check(llama_dart_exit_tracked_count() == tracked_base + 1,
        "the session is not tracked");

  const run_result fixed = run(stream.get(), clip, 0, 0, rate);
  check(fixed.status == LLAMA_DART_ASR_STREAM_STATUS_OK &&
            fixed.progress.state == LLAMA_DART_ASR_STREAM_STATE_FINISHED,
        "the clip did not finish");
  std::printf("TRANSCRIPT<<<%s>>>\n", fixed.text.c_str());
  std::printf("clip %.1fs wall %.2fs rtf %.3f first_token_step %.1fms "
              "steady_max_step %.1fms max_step %.1fms steps %d breaks %d\n",
              static_cast<double>(clip.size()) / rate, fixed.wall_s,
              fixed.wall_s / (static_cast<double>(clip.size()) / rate),
              fixed.first_token_step_ms, fixed.steady_max_step_ms,
              fixed.max_step_ms, fixed.steps, fixed.progress.segment_breaks);
  check(normalize(fixed.text) == kExpected, "the transcript is not the speech");
  llama_dart_asr_stream_progress finished{};
  finished.struct_size = sizeof(finished);
  check(llama_dart_asr_stream_step(stream.get(), &finished) ==
            LLAMA_DART_ASR_STREAM_STATUS_INVALID_STATE,
        "a finished session stepped");

  // Reset, then five random chunkings give the same text.
  for (unsigned seed = 1; seed <= 5; ++seed) {
    check(llama_dart_asr_stream_reset(stream.get()) ==
              LLAMA_DART_ASR_STREAM_STATUS_OK,
          "reset");
    const run_result random = run(stream.get(), clip, 2, seed, rate);
    check(random.text == fixed.text, "a random chunking changed the text");
  }
  std::printf("5 random chunkings and 5 resets: same transcript\n");
  // After the model pages are all touched, so that lazy mapping is not growth.
  const double rss_clip = peak_rss_mb();

  // The whole input pushed in one call keeps every step bounded: upstream
  // alone encodes such a backlog in one call (2.7 s for 297 s of LFM3-Audio).
  const std::vector<float> long_pcm =
      repeated(clip, static_cast<size_t>(opt.long_seconds * rate));
  const std::vector<float> long_input =
      token_delay
          ? repeated(clip,
                     std::min(long_pcm.size(),
                              static_cast<size_t>(info.max_session_ms) * rate /
                                      1000 -
                                  rate))
          : long_pcm;
  llama_dart_asr_stream_params bulk_params = llama_dart_asr_stream_params_default();
  bulk_params.max_buffered_ms =
      static_cast<int32_t>(long_input.size() * 1000 / rate) + 1000;
  stream_ptr bulk_stream = open_stream(context, mtmd, &bulk_params);
  const run_result bulk = run(bulk_stream.get(), long_input, 1, 0, rate);
  bulk_stream.reset();
  check(bulk.status == LLAMA_DART_ASR_STREAM_STATUS_OK, "the long input failed");
  check(bulk.progress.samples_consumed ==
            static_cast<int64_t>(long_input.size()),
        "the long run ended before consuming all input");
  const double rss_long = peak_rss_mb();
  std::printf("long %.0fs wall %.1fs rtf %.3f max_step %.1fms breaks %d "
              "peak_rss %.0f -> %.0f MB\n",
              static_cast<double>(long_input.size()) / rate, bulk.wall_s,
              bulk.wall_s / (static_cast<double>(long_input.size()) / rate),
              bulk.max_step_ms, bulk.progress.segment_breaks, rss_clip, rss_long);
  check(bulk.max_step_ms < opt.max_step_ms, "a step ran past the bound");
  check(rss_long - rss_clip <= 100.0, "the long input grew memory");

  // 44 s: LFM3 restarts its sequence once and loses a word at the cut, so the
  // sentence is intact in the repetitions before and after it. A restart that
  // keeps the old sequence garbles the text after it.
  const std::vector<float> forty_four = repeated(clip, 4 * clip.size());
  check(llama_dart_asr_stream_reset(stream.get()) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "reset");
#if defined(LLAMADART_TESTING)
  const char marker[] = "[!#]";
  const std::string old_tail = std::string(marker) + "\xE2\x82";
  llama_dart_asr_test_hooks hooks;
  hooks.old_segment_tail = old_tail;
  hooks.new_segment_head = "\xAC";
  llama_dart_test_asr_set_hooks(&hooks);
#endif
  const run_result restarts = run(stream.get(), forty_four, 0, 0, rate);
#if defined(LLAMADART_TESTING)
  llama_dart_test_asr_set_hooks(nullptr);
  if (!token_delay) {
    check(restarts.text.find(std::string(marker) + "\xEF\xBF\xBD") !=
              std::string::npos,
          "segment break did not preserve complete text and discard its UTF-8 "
          "tail");
  }
#endif
  check(restarts.status == LLAMA_DART_ASR_STREAM_STATUS_OK, "44 s input failed");
  const std::string restart_text = normalize(restarts.text);
  int intact = 0;
  for (size_t at = restart_text.find(kExpected); at != std::string::npos;
       at = restart_text.find(kExpected, at + 1)) {
    ++intact;
  }
  std::printf("44 s: breaks %d, %d of 4 sentences intact\n",
              restarts.progress.segment_breaks, intact);
  std::printf("TRANSCRIPT44<<<%s>>>\n", restarts.text.c_str());
  check(restarts.progress.segment_breaks == (token_delay ? 0 : 1),
        "unexpected segment breaks for 44 s");
  check(intact >= 3, "the text after a restart is garbled");

  // A cancel lands at the next step, ends the text and releases the stream.
  check(llama_dart_asr_stream_reset(stream.get()) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "reset");
  std::atomic<int8_t> flag{0};
  static_assert(sizeof(std::atomic<int8_t>) == sizeof(int8_t),
                "the cancel flag must be a plain byte");
  check(llama_dart_asr_stream_set_cancel_flag(
            stream.get(), reinterpret_cast<const int8_t *>(&flag)) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "set_cancel_flag");
  size_t accepted = 0;
  check(llama_dart_asr_stream_push(stream.get(), clip.data(), clip.size(),
                                   &accepted) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "push");
  llama_dart_asr_stream_progress progress{};
  progress.struct_size = sizeof(progress);
  int32_t steps_before_cancel = 0;
  while (progress.tokens_sampled < 4 && steps_before_cancel < 2000) {
    check(llama_dart_asr_stream_step(stream.get(), &progress) ==
              LLAMA_DART_ASR_STREAM_STATUS_OK,
          "step before cancel");
    ++steps_before_cancel;
  }
  flag.store(1);
  check(llama_dart_asr_stream_step(stream.get(), &progress) ==
                LLAMA_DART_ASR_STREAM_STATUS_CANCELLED &&
            progress.state == LLAMA_DART_ASR_STREAM_STATE_CANCELLED,
        "the cancel did not land");
  size_t unread = 0;
  check(llama_dart_asr_stream_read_text(stream.get(), nullptr, 0, &unread) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "text query after cancel");
  const int32_t tokens_at_cancel = progress.tokens_sampled;
  check(llama_dart_asr_stream_step(stream.get(), &progress) ==
                LLAMA_DART_ASR_STREAM_STATUS_CANCELLED &&
            progress.tokens_sampled == tokens_at_cancel &&
            progress.text_available == static_cast<int64_t>(unread),
        "text grew after the cancel");
  check(llama_dart_asr_stream_push(stream.get(), clip.data(), 1, &accepted) ==
            LLAMA_DART_ASR_STREAM_STATUS_INVALID_STATE,
        "a cancelled session took audio");
  std::printf("cancel landed after %d tokens\n", tokens_at_cancel);

  // Reuse after a cancel gives the same transcript.
  check(llama_dart_asr_stream_reset(stream.get()) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "reset");
  const run_result again = run(stream.get(), clip, 0, 0, rate);
  check(again.text == fixed.text, "reuse after a cancel changed the text");

  // The thread-safe cancel releases a session that has not started.
  check(llama_dart_asr_stream_reset(stream.get()) ==
            LLAMA_DART_ASR_STREAM_STATUS_OK,
        "reset");
  llama_dart_asr_stream_cancel(stream.get());
  check(llama_dart_asr_stream_step(stream.get(), &progress) ==
            LLAMA_DART_ASR_STREAM_STATUS_CANCELLED,
        "cancel before the first step");

  // Prompted delay contract: 960 ms is the nominal chunk cadence, whereas
  // first visible text needs mel-window lookahead (> 960 ms).
  if (!token_delay) {
    check(llama_dart_asr_stream_reset(stream.get()) ==
              LLAMA_DART_ASR_STREAM_STATUS_OK,
          "reset for delay check");
    const size_t nominal_samples =
        static_cast<size_t>(info.delay_ms) * rate / 1000;
    size_t accepted_nominal = 0;
    check(llama_dart_asr_stream_push(stream.get(), clip.data(), nominal_samples,
                                     &accepted_nominal) ==
                  LLAMA_DART_ASR_STREAM_STATUS_OK &&
              accepted_nominal == nominal_samples,
          "push nominal delay");
    llama_dart_asr_stream_progress dp{};
    dp.struct_size = sizeof(dp);
    for (int s = 0; s < 200; ++s) {
      check(llama_dart_asr_stream_step(stream.get(), &dp) ==
                LLAMA_DART_ASR_STREAM_STATUS_OK,
            "step nominal delay");
      if (dp.state == LLAMA_DART_ASR_STREAM_STATE_NEEDS_AUDIO) {
        break;
      }
    }
    check(dp.text_available == 0,
          "PROMPTED_CHUNKS produced text at nominal 960 ms without lookahead");
    size_t offset = nominal_samples;
    const size_t probe_step = rate / 10; // 100 ms increments
    int64_t first_token_ms = 0;
    int64_t first_text_ms = 0;
    while (offset < clip.size() && first_text_ms == 0) {
      const size_t n = std::min(probe_step, clip.size() - offset);
      size_t acc = 0;
      check(llama_dart_asr_stream_push(stream.get(), clip.data() + offset, n,
                                       &acc) == LLAMA_DART_ASR_STREAM_STATUS_OK,
            "push lookahead");
      offset += acc;
      for (int s = 0; s < 200; ++s) {
        check(llama_dart_asr_stream_step(stream.get(), &dp) ==
                  LLAMA_DART_ASR_STREAM_STATUS_OK,
              "step lookahead");
        if (dp.tokens_sampled > 0 && first_token_ms == 0) {
          first_token_ms = dp.samples_consumed * 1000 / static_cast<int64_t>(rate);
        }
        if (dp.text_available > 0 && first_text_ms == 0) {
          first_text_ms = dp.samples_consumed * 1000 / static_cast<int64_t>(rate);
          break;
        }
        if (dp.state == LLAMA_DART_ASR_STREAM_STATE_NEEDS_AUDIO) {
          break;
        }
      }
    }
    std::printf("prompted onset: nominal delay=%dms first_token=%lldms "
                "first_text=%lldms\n",
                info.delay_ms, static_cast<long long>(first_token_ms),
                static_cast<long long>(first_text_ms));
    check(first_text_ms > info.delay_ms && first_text_ms <= 1600,
          "unexpected prompted first-text onset");
  }

  // A context that cannot hold the audio ends with a typed status and keeps
  // the text so far (both TOKEN_DELAY and small PROMPTED_CHUNKS contexts).
  stream.reset();
  {
    const int small_ctx = token_delay ? 512 : 256;
    llama_context *small = make_context(small_ctx);
    const uint32_t small_context_tokens = llama_n_ctx_seq(small);
    check(small_context_tokens == static_cast<uint32_t>(small_ctx),
          "effective small context did not match the requested size");
    stream_ptr limited = open_stream(small, mtmd);
    const run_result full = run(limited.get(), forty_four, 0, 0, rate);
    check(full.status == LLAMA_DART_ASR_STREAM_STATUS_CONTEXT_FULL &&
              full.progress.state == LLAMA_DART_ASR_STREAM_STATE_FINISHED &&
              !full.text.empty(),
          "context exhaustion was not a typed status with text");
    std::printf("context full (%u positions per sequence) after %lld tokens, "
                "text kept (%zu bytes)\n",
                small_context_tokens,
                static_cast<long long>(full.progress.tokens_sampled),
                full.text.size());
    limited.reset();
    llama_dart_exit_free(small);
  }

  // Cycles leave nothing tracked.
  for (int i = 0; i < 10; ++i) {
    stream_ptr cycle = open_stream(context, mtmd);
    size_t pushed = 0;
    check(llama_dart_asr_stream_push(cycle.get(), clip.data(), rate, &pushed) ==
              LLAMA_DART_ASR_STREAM_STATUS_OK,
          "cycle push");
    llama_dart_asr_stream_progress cycle_progress{};
    cycle_progress.struct_size = sizeof(cycle_progress);
    for (int step = 0; step < 6; ++step) {
      check(llama_dart_asr_stream_step(cycle.get(), &cycle_progress) ==
                LLAMA_DART_ASR_STREAM_STATUS_OK,
            "cycle step");
    }
    if (i % 2 == 0) {
      llama_dart_asr_stream_cancel(cycle.get());
    }
  }
  check(llama_dart_exit_tracked_count() == tracked_base,
        "sessions stayed tracked");

  llama_dart_exit_free(mtmd);
  llama_dart_exit_free(context);
  llama_dart_exit_free(model);
  check(llama_dart_exit_tracked_count() == tracked_start, "objects leaked");
  llama_backend_free();
  std::printf("PASS mode=%s backend=%s\n",
              token_delay ? "TOKEN_DELAY" : "PROMPTED_CHUNKS",
              opt.gpu ? "gpu" : "cpu");
  return 0;
}
