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
#include <limits>
#include <memory>
#include <string>
#include <vector>

using steady_clock = std::chrono::steady_clock;

using tts_ptr = std::unique_ptr<llama_dart_tts, decltype(&llama_dart_tts_free)>;

static constexpr uint32_t qwen3_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
static constexpr uint32_t lfm3_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;
static constexpr uint32_t soprano_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_SOPRANO;

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

struct llama_backend_guard {
    llama_backend_guard() { llama_backend_init(); }
    ~llama_backend_guard() { llama_backend_free(); }
};

static bool fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    return false;
}

static llama_dart_tts_status run_to_end(llama_dart_tts *tts, llama_dart_tts_progress *progress,
                                        double *last_step_ms) {
    for (;;) {
        progress->struct_size = sizeof(*progress);
        const auto started = steady_clock::now();
        const llama_dart_tts_status status = llama_dart_tts_step(tts, progress);
        *last_step_ms =
            std::chrono::duration<double, std::milli>(steady_clock::now() - started).count();
        if (status != LLAMA_DART_TTS_STATUS_OK ||
            progress->state == LLAMA_DART_TTS_STATE_COMPLETED) {
            return status;
        }
    }
}

static bool synthesize(llama_dart_tts *tts, const llama_dart_tts_request &request,
                       const llama_dart_tts_info &info, std::vector<float> *pcm,
                       llama_dart_tts_progress *progress, double *decode_ms) {
    if (llama_dart_tts_start(tts, &request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start TTS: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    if (run_to_end(tts, progress, decode_ms) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "TTS failed: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    llama_dart_tts_output_info output{};
    output.struct_size = sizeof(output);
    size_t count = 0;
    if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK ||
        output.sample_rate != info.sample_rate || output.channels != 1 ||
        output.sample_count <= 0 ||
        llama_dart_tts_read_pcm(tts, 0, nullptr, 0, &count) != LLAMA_DART_TTS_STATUS_OK ||
        count != static_cast<size_t>(output.sample_count)) {
        return fail("invalid TTS output metadata");
    }
    pcm->resize(count);
    size_t read = 0;
    if (llama_dart_tts_read_pcm(tts, 0, pcm->data(), pcm->size(), &read) !=
            LLAMA_DART_TTS_STATUS_OK ||
        read != count) {
        return fail("failed to read PCM output");
    }
    double energy = 0.0;
    for (float sample : *pcm) {
        if (!std::isfinite(sample)) {
            return fail("PCM contains non-finite samples");
        }
        energy += static_cast<double>(sample) * sample;
    }
    if (std::sqrt(energy / pcm->size()) < 1e-5) {
        return fail("PCM output is effectively silent");
    }
    return true;
}

static bool same_pcm(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() &&
           std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Soprano's frame steps run only the backbone, so the first mtmd graph node of
// a task is in its audio decode. The probe cancels there and counts the chunk
// ends that follow: the decode must stop at its first one.
static std::atomic<llama_dart_tts *> cancel_at_decode{nullptr};
static std::atomic<int> chunk_ends_after_cancel{0};
static std::atomic<bool> cancel_issued{false};

static bool probe_eval_callback(struct ggml_tensor *tensor, bool ask, void *user_data) {
    if (ask) {
        if (llama_dart_tts *tts = cancel_at_decode.exchange(nullptr)) {
            llama_dart_tts_cancel(tts);
            cancel_issued.store(true);
        }
    } else if (cancel_issued.load()) {
        chunk_ends_after_cancel.fetch_add(1);
    }
    return llama_dart_tts_eval_callback(tensor, ask, user_data);
}

static bool cancel_during_decode(llama_dart_tts *tts, const llama_dart_tts_request &request,
                                 double *cancel_step_ms) {
    if (llama_dart_tts_start(tts, &request) != LLAMA_DART_TTS_STATUS_OK) {
        return fail("failed to start the in-decode cancel probe");
    }
    chunk_ends_after_cancel.store(0);
    cancel_issued.store(false);
    cancel_at_decode.store(tts);
    llama_dart_tts_progress progress{};
    const llama_dart_tts_status status = run_to_end(tts, &progress, cancel_step_ms);
    cancel_at_decode.store(nullptr);
    if (llama_dart_tts_reset(tts) != LLAMA_DART_TTS_STATUS_OK) {
        return fail("reset after the in-decode cancel probe failed");
    }
    if (!cancel_issued.load() || status != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED || progress.frames_generated < 2) {
        std::fprintf(stderr, "in-decode cancel: issued=%d status=%d state=%d frames=%d\n",
                     static_cast<int>(cancel_issued.load()), static_cast<int>(status),
                     static_cast<int>(progress.state), progress.frames_generated);
        return false;
    }
    if (chunk_ends_after_cancel.load() > 1) {
        std::fprintf(stderr, "the decode ran %d chunks after the cancel\n",
                     chunk_ends_after_cancel.load());
        return false;
    }
    return true;
}

// Callers that did not opt into SOPRANO, including every
// llama_dart_tts_get_info and llama_dart_tts_init caller, see UNKNOWN.
static bool check_soprano_needs_opt_in(mtmd_context *mtmd, llama_context *context) {
    for (int ext = 0; ext < 2; ++ext) {
        llama_dart_tts_info info{};
        info.struct_size = sizeof(info);
        llama_dart_tts_status status =
            ext ? llama_dart_tts_get_info_ext(mtmd, qwen3_bit | lfm3_bit, &info)
                : llama_dart_tts_get_info(mtmd, &info);
        if (status != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
            info.model_type != LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN || info.sample_rate != 0 ||
            info.channels != 0 || info.capabilities != 0) {
            return fail("a caller without the Soprano bit saw a Soprano projector");
        }
        status = LLAMA_DART_TTS_STATUS_OK;
        tts_ptr tts(ext ? llama_dart_tts_init_ext(context, mtmd, qwen3_bit | lfm3_bit, &status)
                        : llama_dart_tts_init(context, mtmd, &status),
                    llama_dart_tts_free);
        if (tts != nullptr || status != LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
            return fail("a caller without the Soprano bit created a Soprano task");
        }
    }
    return true;
}

static bool check_rejected_request(llama_dart_tts *tts, const llama_dart_tts_request &request,
                                   const char *what) {
    if (llama_dart_tts_start(tts, &request) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        std::strstr(llama_dart_tts_last_error(tts), what) == nullptr) {
        std::fprintf(stderr, "a Soprano request with a %s was not rejected\n", what);
        return false;
    }
    const int8_t flag = 0;
    if (llama_dart_tts_set_cancel_flag(tts, &flag) != LLAMA_DART_TTS_STATUS_INVALID_STATE) {
        std::fprintf(stderr, "a rejected %s request started a task\n", what);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const bool use_gpu = argc > 1 && std::strcmp(argv[argc - 1], "--gpu") == 0;
    const int value_argc = argc - (use_gpu ? 1 : 0);
    if (value_argc < 4 || value_argc > 5) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ OUTPUT_WAV [TEXT] [--gpu]\n", argv[0]);
        return 2;
    }
    const char *text = value_argc >= 5 ? argv[4] : "Hello from Llama Dart.";
    if ((llama_dart_tts_supported_model_types() & soprano_bit) == 0) {
        std::fprintf(stderr, "this libllamadart was built without Soprano\n");
        return 1;
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
    mtmd_params.cb_eval = probe_eval_callback;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
        mtmd_init_from_file(argv[2], model.get(), mtmd_params), mtmd_free);
    if (mtmd == nullptr) {
        std::fprintf(stderr, "failed to load mmproj\n");
        return 1;
    }
    if (!check_soprano_needs_opt_in(mtmd.get(), context.get())) {
        return 1;
    }

    llama_dart_tts_info info{};
    info.struct_size = sizeof(info);
    if (llama_dart_tts_get_info_ext(mtmd.get(), qwen3_bit | soprano_bit, &info) !=
            LLAMA_DART_TTS_STATUS_OK ||
        info.model_type != LLAMA_DART_TTS_MODEL_TYPE_SOPRANO ||
        info.api_version != LLAMA_DART_TTS_API_VERSION || info.sample_rate <= 0 ||
        info.channels != 1 || info.capabilities != 0) {
        std::fprintf(stderr, "invalid Soprano capability metadata\n");
        return 1;
    }
    llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    tts_ptr tts(llama_dart_tts_init_ext(context.get(), mtmd.get(), qwen3_bit | soprano_bit, &status),
                llama_dart_tts_free);
    if (tts == nullptr) {
        std::fprintf(stderr, "failed to initialize Soprano TTS: %d\n", static_cast<int>(status));
        return 1;
    }

    llama_dart_tts_request request = llama_dart_tts_request_default();
    request.text = text;
    request.text_length = std::strlen(text);
    request.seed = 1;
    const unsigned char speaker[] = {0};
    llama_dart_tts_request with_speaker = request;
    with_speaker.speaker_audio = speaker;
    with_speaker.speaker_audio_length = sizeof(speaker);
    llama_dart_tts_request with_language = request;
    with_language.language = "en";
    if (!check_rejected_request(tts.get(), with_speaker, "speaker") ||
        !check_rejected_request(tts.get(), with_language, "language")) {
        return 1;
    }
    if (llama_dart_tts_start_with_voice(tts.get(), &request, "female") !=
        LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
        std::fprintf(stderr, "a Soprano request with a voice was not rejected\n");
        return 1;
    }

    if (llama_dart_tts_start(tts.get(), &request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start the cancel probe\n");
        return 1;
    }
    llama_dart_tts_cancel(tts.get());
    llama_dart_tts_progress progress{};
    progress.struct_size = sizeof(progress);
    if (llama_dart_tts_step(tts.get(), &progress) != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED ||
        llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "cancel/reset probe failed\n");
        return 1;
    }

    llama_dart_tts_request one_frame = request;
    one_frame.max_frames = 1;
    double unused_ms = 0.0;
    if (llama_dart_tts_start(tts.get(), &one_frame) != LLAMA_DART_TTS_STATUS_OK ||
        run_to_end(tts.get(), &progress, &unused_ms) != LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR ||
        progress.state != LLAMA_DART_TTS_STATE_FAILED) {
        std::fprintf(stderr, "a one-frame Soprano utterance did not fail\n");
        return 1;
    }

    std::vector<float> first;
    llama_dart_tts_progress first_progress{};
    double decode_ms = 0.0;
    if (!synthesize(tts.get(), request, info, &first, &first_progress, &decode_ms)) {
        return 1;
    }
    double cancel_ms = 0.0;
    if (!cancel_during_decode(tts.get(), request, &cancel_ms)) {
        return 1;
    }
    llama_dart_tts_request repeat = request;
    repeat.sequence_id = 1;
    std::vector<float> second;
    llama_dart_tts_progress second_progress{};
    double second_decode_ms = 0.0;
    if (!synthesize(tts.get(), repeat, info, &second, &second_progress, &second_decode_ms)) {
        return 1;
    }
    if (!same_pcm(first, second)) {
        std::fprintf(stderr, "a repeated seed did not repeat the PCM\n");
        return 1;
    }
    llama_dart_tts_request other_seed = request;
    other_seed.seed = 2;
    std::vector<float> third;
    llama_dart_tts_progress third_progress{};
    if (!synthesize(tts.get(), other_seed, info, &third, &third_progress, &second_decode_ms)) {
        return 1;
    }
    if (same_pcm(first, third)) {
        std::fprintf(stderr, "the seed did not reach the sampler\n");
        return 1;
    }

    llama_dart_tts_request truncated = request;
    truncated.max_frames = 2;
    std::vector<float> short_pcm;
    llama_dart_tts_progress short_progress{};
    if (!synthesize(tts.get(), truncated, info, &short_pcm, &short_progress, &second_decode_ms) ||
        !short_progress.truncated || short_progress.frames_generated != 2) {
        std::fprintf(stderr, "max_frames did not truncate the utterance\n");
        return 1;
    }
    if (llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK ||
        !write_wav(argv[3], first, info.sample_rate)) {
        std::fprintf(stderr, "failed to reset or write WAV output\n");
        return 1;
    }
    std::printf("PASS backend=%s model_type=%d sample_rate=%d samples=%zu frames=%d "
                "decode_ms=%.1f cancel_step_ms=%.1f truncated_samples=%zu\n",
                use_gpu ? "gpu" : "cpu", static_cast<int>(info.model_type), info.sample_rate,
                first.size(), first_progress.frames_generated, decode_ms, cancel_ms,
                short_pcm.size());
    return 0;
}
