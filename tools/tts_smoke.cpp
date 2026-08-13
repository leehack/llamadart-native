#include "llama_dart_wrapper.h"

#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

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
                       double *rms) {
    if (llama_dart_tts_start(tts, request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start TTS: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    progress->struct_size = sizeof(*progress);
    do {
        const llama_dart_tts_status status = llama_dart_tts_step(tts, progress);
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

int main(int argc, char **argv) {
    if (argc < 4 || argc > 7) {
        std::fprintf(stderr,
                     "usage: %s MODEL MMPROJ OUTPUT_WAV [TEXT] [LANGUAGE] [SPEAKER_AUDIO]\n",
                     argv[0]);
        return 2;
    }
    const char *text = argc >= 5 ? argv[4] : "Hello from Llama Dart.";
    const char *language = argc >= 6 ? argv[5] : "en";
    const std::vector<unsigned char> speaker = argc >= 7 ? read_file(argv[6]) : std::vector<unsigned char>();
    if (argc >= 7 && speaker.empty()) {
        std::fprintf(stderr, "failed to read speaker audio\n");
        return 2;
    }

    const llama_backend_guard backend;
    llama_dart_set_log_level(4);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
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
    mtmd_params.use_gpu = false;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
        mtmd_init_from_file(argv[2], model.get(), mtmd_params), mtmd_free);
    if (mtmd == nullptr) {
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
    if (tts == nullptr) {
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

    std::vector<float> first_pcm;
    llama_dart_tts_progress first_progress{};
    double first_rms = 0.0;
    if (!synthesize(tts.get(), &request, info, &first_pcm, &first_progress, &first_rms)) {
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
    std::printf("PASS model_type=%d sample_rate=%d first_samples=%zu second_samples=%zu "
                "first_frames=%d second_frames=%d first_rms=%.6f second_rms=%.6f\n",
                static_cast<int>(info.model_type), info.sample_rate, first_pcm.size(),
                second_pcm.size(), first_progress.frames_generated,
                second_progress.frames_generated, first_rms, second_rms);
    return 0;
}
