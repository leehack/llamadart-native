#include "llama_dart_wrapper.h"

#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using steady_clock = std::chrono::steady_clock;

static const char *const long_text =
    "Hello from Llama Dart. This sentence is here only to make the speech long "
    "enough that the decoder buffers a full window of codec frames before the "
    "end, so the flush runs in the middle of synthesis.";

static std::vector<steady_clock::time_point> chunk_ends;

static bool recording_eval_callback(struct ggml_tensor *tensor, bool ask, void *user_data) {
    const bool answer = llama_dart_tts_eval_callback(tensor, ask, user_data);
    if (!ask) {
        chunk_ends.push_back(steady_clock::now());
    }
    return answer;
}

struct step_trace {
    double ms;
    int chunk_boundaries;
    double longest_chunk_ms;
};

static double elapsed_ms(steady_clock::time_point from, steady_clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

static void write_u16(std::ofstream &out, uint16_t value) {
    const char bytes[] = {static_cast<char>(value), static_cast<char>(value >> 8)};
    out.write(bytes, sizeof(bytes));
}

static void write_u32(std::ofstream &out, uint32_t value) {
    const char bytes[] = {
        static_cast<char>(value),
        static_cast<char>(value >> 8),
        static_cast<char>(value >> 16),
        static_cast<char>(value >> 24),
    };
    out.write(bytes, sizeof(bytes));
}

static bool write_wav(const std::string &path, const std::vector<float> &pcm, int32_t sample_rate) {
    if (pcm.size() > (std::numeric_limits<uint32_t>::max() - 44) / 2) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    out.write("RIFF", 4);
    write_u32(out, 36 + data_size);
    out.write("WAVEfmt ", 8);
    write_u32(out, 16);
    write_u16(out, 1);
    write_u16(out, 1);
    write_u32(out, static_cast<uint32_t>(sample_rate));
    write_u32(out, static_cast<uint32_t>(sample_rate * 2));
    write_u16(out, 2);
    write_u16(out, 16);
    out.write("data", 4);
    write_u32(out, data_size);
    for (float sample : pcm) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const int16_t value = static_cast<int16_t>(std::lrintf(clipped * 32767.0f));
        write_u16(out, static_cast<uint16_t>(value));
    }
    return static_cast<bool>(out);
}

static std::vector<unsigned char> read_file(const char *path) {
    if (path == nullptr) {
        return {};
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return {};
    }
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(input), {});
}

struct llama_backend_guard {
    llama_backend_guard() { llama_backend_init(); }
    ~llama_backend_guard() { llama_backend_free(); }
};

static bool synthesize(llama_dart_tts *tts,
                       llama_dart_tts_request *request,
                       const llama_dart_tts_info &info,
                       std::vector<float> *pcm,
                       llama_dart_tts_progress *progress,
                       double *rms,
                       std::vector<step_trace> *trace = nullptr) {
    if (llama_dart_tts_start(tts, request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start TTS: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    progress->struct_size = sizeof(*progress);
    do {
        chunk_ends.clear();
        const auto started = steady_clock::now();
        const llama_dart_tts_status status = llama_dart_tts_step(tts, progress);
        const auto returned = steady_clock::now();
        if (trace != nullptr) {
            double longest = 0.0;
            auto previous = started;
            chunk_ends.push_back(returned);
            for (const auto &end : chunk_ends) {
                longest = std::max(longest, elapsed_ms(previous, end));
                previous = end;
            }
            trace->push_back({elapsed_ms(started, returned),
                              static_cast<int>(chunk_ends.size()) - 1, longest});
        }
        if (status != LLAMA_DART_TTS_STATUS_OK) {
            std::fprintf(stderr, "TTS failed: %s\n", llama_dart_tts_last_error(tts));
            return false;
        }
    } while (progress->state != LLAMA_DART_TTS_STATE_COMPLETED);

    llama_dart_tts_output_info output{};
    output.struct_size = sizeof(output);
    if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK ||
        output.sample_rate != info.sample_rate || output.channels != 1 || output.sample_count <= 0) {
        std::fprintf(stderr, "invalid TTS output metadata\n");
        return false;
    }
    size_t remaining = 0;
    if (llama_dart_tts_read_pcm(tts, 0, nullptr, 0, &remaining) !=
            LLAMA_DART_TTS_STATUS_OK ||
        remaining != static_cast<size_t>(output.sample_count)) {
        std::fprintf(stderr, "failed to query PCM output length\n");
        return false;
    }
    pcm->resize(remaining);
    size_t offset = 0;
    while (offset < pcm->size()) {
        size_t read = 0;
        const size_t capacity = std::min<size_t>(4096, pcm->size() - offset);
        if (llama_dart_tts_read_pcm(tts, static_cast<int64_t>(offset), pcm->data() + offset,
                                    capacity, &read) != LLAMA_DART_TTS_STATUS_OK ||
            read != capacity) {
            std::fprintf(stderr, "failed to read PCM output chunk\n");
            return false;
        }
        offset += read;
    }
    double energy = 0.0;
    for (float sample : *pcm) {
        if (!std::isfinite(sample)) {
            std::fprintf(stderr, "PCM contains non-finite samples\n");
            return false;
        }
        energy += static_cast<double>(sample) * sample;
    }
    *rms = std::sqrt(energy / pcm->size());
    if (*rms < 1e-5) {
        std::fprintf(stderr, "PCM output is effectively silent\n");
        return false;
    }
    return true;
}

static bool cancel_during_long_step(llama_dart_tts *tts,
                                    llama_dart_tts_request *request,
                                    std::atomic<int8_t> *flag,
                                    double cancel_after_ms,
                                    double return_within_ms,
                                    double *cancel_to_return_ms,
                                    int *frames_before_cancel) {
    static_assert(sizeof(std::atomic<int8_t>) == sizeof(int8_t),
                  "the cancel flag must be a plain byte");
    if (llama_dart_tts_start(tts, request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start in-step cancel probe: %s\n",
                     llama_dart_tts_last_error(tts));
        return false;
    }
    if (flag != nullptr &&
        llama_dart_tts_set_cancel_flag(tts, reinterpret_cast<const int8_t *>(flag)) !=
            LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to attach cancel flag\n");
        return false;
    }
    std::atomic<int> running_step{-1};
    std::atomic<int> running_after_frames{0};
    std::atomic<int64_t> running_since{0};
    std::atomic<int> cancelled_step{-1};
    std::atomic<int64_t> cancelled_at{0};
    std::atomic<bool> done{false};
    const auto origin = steady_clock::now();
    auto since_origin = [&origin] {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   steady_clock::now() - origin)
            .count();
    };
    std::thread canceller([&] {
        while (!done.load()) {
            const int step = running_step.load();
            if (step >= 0 && running_after_frames.load() > 0 &&
                since_origin() - running_since.load() >= cancel_after_ms * 1000.0) {
                cancelled_at.store(since_origin());
                cancelled_step.store(step);
                if (flag != nullptr) {
                    flag->store(1);
                } else {
                    llama_dart_tts_cancel(tts);
                }
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    llama_dart_tts_progress progress{};
    llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    int step = 0;
    int64_t returned_at = 0;
    for (;; ++step) {
        const int frames = progress.frames_generated;
        progress.struct_size = sizeof(progress);
        running_since.store(since_origin());
        running_after_frames.store(frames);
        running_step.store(step);
        status = llama_dart_tts_step(tts, &progress);
        running_step.store(-1);
        returned_at = since_origin();
        if (status != LLAMA_DART_TTS_STATUS_OK ||
            progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            break;
        }
    }
    done.store(true);
    canceller.join();
    const bool same_step = cancelled_step.load() == step;
    *cancel_to_return_ms = (returned_at - cancelled_at.load()) / 1000.0;
    *frames_before_cancel = progress.frames_generated;
    if (llama_dart_tts_reset(tts) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "reset after in-step cancel probe failed\n");
        return false;
    }
    if (cancelled_step.load() < 0) {
        std::fprintf(stderr, "no step after a frame ran %.1f ms, so the in-step cancel never fired\n",
                     cancel_after_ms);
        return false;
    }
    if (!same_step || status != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED) {
        std::fprintf(stderr,
                     "in-step cancel at step %d ended at step %d with status %d state %d\n",
                     cancelled_step.load(), step, static_cast<int>(status),
                     static_cast<int>(progress.state));
        return false;
    }
    if (*cancel_to_return_ms > return_within_ms) {
        std::fprintf(stderr, "in-step cancel took %.1f ms to return, bound %.1f ms\n",
                     *cancel_to_return_ms, return_within_ms);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const bool use_gpu = argc > 1 && std::strcmp(argv[argc - 1], "--gpu") == 0;
    const int value_argc = argc - (use_gpu ? 1 : 0);
    if (value_argc < 4 || value_argc > 7) {
        std::fprintf(stderr,
                     "usage: %s MODEL MMPROJ OUTPUT_WAV [TEXT] [LANGUAGE] "
                     "[SPEAKER_AUDIO] [--gpu]\n",
                     argv[0]);
        return 2;
    }
    const char *text = value_argc >= 5 ? argv[4] : "Hello from Llama Dart.";
    const char *language = value_argc >= 6 ? argv[5] : "en";
    const std::vector<unsigned char> speaker =
        value_argc >= 7 ? read_file(argv[6]) : std::vector<unsigned char>();
    if (value_argc >= 7 && speaker.empty()) {
        std::fprintf(stderr, "failed to read speaker audio\n");
        return 2;
    }

    const llama_backend_guard backend;
    llama_dart_set_log_level(4);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? 99 : 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(argv[1], model_params), llama_model_free);
    if (model == nullptr) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }
    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 4096;
    context_params.n_batch = 512;
    context_params.n_ubatch = 512;
    context_params.n_seq_max = 2;
    context_params.embeddings = true;
    std::unique_ptr<llama_context, decltype(&llama_free)> context(
        llama_init_from_model(model.get(), context_params), llama_free);
    if (context == nullptr) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> plain_mtmd(
        mtmd_init_from_file(argv[2], model.get(), mtmd_params), mtmd_free);
    mtmd_params.cb_eval = recording_eval_callback;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
        mtmd_init_from_file(argv[2], model.get(), mtmd_params), mtmd_free);
    if (plain_mtmd == nullptr || mtmd == nullptr) {
        std::fprintf(stderr, "failed to load mmproj\n");
        return 1;
    }

    llama_dart_tts_info info{};
    info.struct_size = sizeof(info);
    if (llama_dart_tts_get_info(mtmd.get(), &info) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "mmproj does not support TTS\n");
        return 1;
    }
    if (info.api_version != LLAMA_DART_TTS_API_VERSION || info.sample_rate <= 0 || info.channels != 1) {
        std::fprintf(stderr, "invalid TTS capability metadata\n");
        return 1;
    }
    if ((info.capabilities & LLAMA_DART_TTS_CAPABILITY_LANGUAGE) == 0 ||
        (!speaker.empty() &&
         (info.capabilities & LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE) == 0)) {
        std::fprintf(stderr, "missing required TTS capability metadata\n");
        return 1;
    }
    llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    std::unique_ptr<llama_dart_tts, decltype(&llama_dart_tts_free)> tts(
        llama_dart_tts_init(context.get(), mtmd.get(), &status), llama_dart_tts_free);
    std::unique_ptr<llama_dart_tts, decltype(&llama_dart_tts_free)> plain_tts(
        llama_dart_tts_init(context.get(), plain_mtmd.get(), &status), llama_dart_tts_free);
    if (tts == nullptr || plain_tts == nullptr) {
        std::fprintf(stderr, "failed to initialize TTS wrapper: %d\n", static_cast<int>(status));
        return 1;
    }
    llama_dart_tts_request request = llama_dart_tts_request_default();
    request.text = text;
    request.text_length = std::char_traits<char>::length(text);
    request.language = language;
    request.seed = 1;
    request.speaker_audio = speaker.empty() ? nullptr : speaker.data();
    request.speaker_audio_length = speaker.size();
    llama_dart_tts_request invalid_request = request;
    invalid_request.top_p = std::numeric_limits<float>::quiet_NaN();
    if (llama_dart_tts_start(tts.get(), &invalid_request) !=
        LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT) {
        std::fprintf(stderr, "non-finite sampling validation failed\n");
        return 1;
    }
    std::atomic<int8_t> cancel_flag{0};
    const int8_t *cancel_flag_address = reinterpret_cast<const int8_t *>(&cancel_flag);
    if (llama_dart_tts_set_cancel_flag(tts.get(), cancel_flag_address) !=
            LLAMA_DART_TTS_STATUS_INVALID_STATE ||
        llama_dart_tts_set_cancel_flag(tts.get(), nullptr) !=
            LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT) {
        std::fprintf(stderr, "cancel flag validation failed\n");
        return 1;
    }
    if (llama_dart_tts_start(tts.get(), &request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start cancellation probe: %s\n",
                     llama_dart_tts_last_error(tts.get()));
        return 1;
    }
    llama_dart_tts_cancel(tts.get());
    llama_dart_tts_progress cancelled{};
    cancelled.struct_size = sizeof(cancelled);
    if (llama_dart_tts_step(tts.get(), &cancelled) != LLAMA_DART_TTS_STATUS_CANCELLED ||
        cancelled.state != LLAMA_DART_TTS_STATE_CANCELLED ||
        llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "TTS cancellation/reset probe failed\n");
        return 1;
    }

    std::vector<float> plain_pcm;
    llama_dart_tts_progress plain_progress{};
    double plain_rms = 0.0;
    if (!synthesize(plain_tts.get(), &request, info, &plain_pcm, &plain_progress, &plain_rms)) {
        return 1;
    }

    std::vector<float> first_pcm;
    llama_dart_tts_progress first_progress{};
    double first_rms = 0.0;
    std::vector<step_trace> first_trace;
    if (!synthesize(tts.get(), &request, info, &first_pcm, &first_progress, &first_rms,
                    &first_trace)) {
        return 1;
    }
    if (first_pcm.size() != plain_pcm.size() ||
        std::memcmp(first_pcm.data(), plain_pcm.data(), first_pcm.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "the eval callback changed uncancelled PCM\n");
        return 1;
    }
    const double decode_ms = first_trace.back().ms;
    for (const step_trace &trace : first_trace) {
        if (trace.ms < decode_ms / 4 && trace.chunk_boundaries != 0) {
            std::fprintf(stderr, "a %.1f ms frame step was split into chunks\n", trace.ms);
            return 1;
        }
    }
    if (first_trace.back().longest_chunk_ms > decode_ms / 4) {
        std::fprintf(stderr, "a %.1f ms chunk of the %.1f ms final audio decode exceeds a quarter\n",
                     first_trace.back().longest_chunk_ms, decode_ms);
        return 1;
    }

    double final_cancel_ms = 0.0;
    int final_cancel_frames = 0;
    if (!cancel_during_long_step(tts.get(), &request, nullptr, decode_ms / 4, decode_ms / 3,
                                 &final_cancel_ms, &final_cancel_frames)) {
        return 1;
    }
    llama_dart_tts_request long_request = request;
    long_request.text = long_text;
    long_request.text_length = std::char_traits<char>::length(long_text);
    double window_cancel_ms = 0.0;
    int window_cancel_frames = 0;
    if (!cancel_during_long_step(tts.get(), &long_request, &cancel_flag, decode_ms / 4,
                                 decode_ms / 3, &window_cancel_ms, &window_cancel_frames)) {
        return 1;
    }

    request.sequence_id = 1;
    std::vector<float> second_pcm;
    llama_dart_tts_progress second_progress{};
    double second_rms = 0.0;
    if (!synthesize(tts.get(), &request, info, &second_pcm, &second_progress, &second_rms)) {
        return 1;
    }
    if (llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "TTS final reset failed\n");
        return 1;
    }
    if (!write_wav(argv[3], second_pcm, info.sample_rate)) {
        std::fprintf(stderr, "failed to write WAV output\n");
        return 1;
    }
    std::printf("PASS backend=%s model_type=%d sample_rate=%d "
                "first_samples=%zu second_samples=%zu "
                "first_frames=%d second_frames=%d first_rms=%.6f second_rms=%.6f "
                "decode_ms=%.1f decode_chunks=%d longest_chunk_ms=%.1f "
                "final_cancel_ms=%.1f final_cancel_frames=%d "
                "window_cancel_ms=%.1f window_cancel_frames=%d\n",
                use_gpu ? "gpu" : "cpu", static_cast<int>(info.model_type),
                info.sample_rate, first_pcm.size(),
                second_pcm.size(), first_progress.frames_generated,
                second_progress.frames_generated, first_rms, second_rms, decode_ms,
                first_trace.back().chunk_boundaries + 1, first_trace.back().longest_chunk_ms,
                final_cancel_ms, final_cancel_frames,
                window_cancel_ms, window_cancel_frames);
    return 0;
}
