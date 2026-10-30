#include "llama_dart_wrapper.h"

#include "llama.h"
#include "mtmd.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static const char *const speech_text = "Hello, how are you today?";

static constexpr uint32_t qwen3_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
static constexpr uint32_t lfm3_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO;

struct llama_backend_guard {
    llama_backend_guard() { llama_backend_init(); }
    ~llama_backend_guard() { llama_backend_free(); }
};

static void append_u16(std::vector<unsigned char> &out, uint16_t value) {
    out.push_back(static_cast<unsigned char>(value));
    out.push_back(static_cast<unsigned char>(value >> 8));
}

static void append_u32(std::vector<unsigned char> &out, uint32_t value) {
    for (int shift = 0; shift < 32; shift += 8) {
        out.push_back(static_cast<unsigned char>(value >> shift));
    }
}

static std::vector<unsigned char> wav16(const std::vector<float> &pcm, int32_t sample_rate) {
    std::vector<unsigned char> out;
    const uint32_t data_size = static_cast<uint32_t>(pcm.size() * 2);
    const char *riff = "RIFF";
    out.insert(out.end(), riff, riff + 4);
    append_u32(out, 36 + data_size);
    const char *fmt = "WAVEfmt ";
    out.insert(out.end(), fmt, fmt + 8);
    append_u32(out, 16);
    append_u16(out, 1);
    append_u16(out, 1);
    append_u32(out, static_cast<uint32_t>(sample_rate));
    append_u32(out, static_cast<uint32_t>(sample_rate * 2));
    append_u16(out, 2);
    append_u16(out, 16);
    const char *data = "data";
    out.insert(out.end(), data, data + 4);
    append_u32(out, data_size);
    for (float sample : pcm) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        append_u16(out, static_cast<uint16_t>(static_cast<int16_t>(std::lrintf(clipped * 32767.0f))));
    }
    return out;
}

static bool run_to_end(llama_dart_tts *tts, llama_dart_tts_progress *progress) {
    do {
        progress->struct_size = sizeof(*progress);
        if (llama_dart_tts_step(tts, progress) != LLAMA_DART_TTS_STATUS_OK) {
            std::fprintf(stderr, "TTS step failed: %s\n", llama_dart_tts_last_error(tts));
            return false;
        }
    } while (progress->state != LLAMA_DART_TTS_STATE_COMPLETED);
    return true;
}

static bool read_pcm(const llama_dart_tts *tts, int32_t sample_rate, std::vector<float> *pcm) {
    llama_dart_tts_output_info output{};
    output.struct_size = sizeof(output);
    if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK ||
        output.sample_rate != sample_rate || output.channels != 1 || output.sample_count <= 0) {
        std::fprintf(stderr, "invalid TTS output metadata\n");
        return false;
    }
    pcm->resize(static_cast<size_t>(output.sample_count));
    size_t read = 0;
    if (llama_dart_tts_read_pcm(tts, 0, pcm->data(), pcm->size(), &read) != LLAMA_DART_TTS_STATUS_OK ||
        read != pcm->size()) {
        std::fprintf(stderr, "failed to read PCM\n");
        return false;
    }
    return true;
}

static double finite_rms(const std::vector<float> &pcm) {
    double energy = 0.0;
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            return -1.0;
        }
        energy += static_cast<double>(sample) * sample;
    }
    return std::sqrt(energy / pcm.size());
}

static bool synthesize(llama_dart_tts *tts, const llama_dart_tts_request &request, const char *voice,
                       int32_t sample_rate, std::vector<float> *pcm,
                       llama_dart_tts_progress *progress) {
    if (llama_dart_tts_start_with_voice(tts, &request, voice) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start TTS: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    if (!run_to_end(tts, progress) || !read_pcm(tts, sample_rate, pcm)) {
        return false;
    }
    const double rms = finite_rms(*pcm);
    if (rms < 1e-4) {
        std::fprintf(stderr, "PCM is non-finite or silent (rms %.6f)\n", rms);
        return false;
    }
    return true;
}

static bool same_pcm(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// Transcribes speech through the generic mtmd input exports and a hand-written
// prompt of the model card's ASR system message.
static std::string transcribe(llama_context *context, mtmd_context *mtmd, const llama_model *model,
                              const std::vector<unsigned char> &wav) {
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
        llama_dart_mtmd_bitmap_init_from_buf(mtmd, wav.data(), wav.size()), mtmd_bitmap_free);
    if (bitmap == nullptr || !mtmd_bitmap_is_audio(bitmap.get())) {
        std::fprintf(stderr, "failed to decode the synthesized WAV as audio input\n");
        return {};
    }
    const std::string prompt = std::string("<|im_start|>system\nPerform ASR in English.<|im_end|>\n"
                                           "<|im_start|>user\n") +
                               mtmd_default_marker() + "<|im_end|>\n<|im_start|>assistant\n";
    mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free);
    const mtmd_bitmap *bitmaps[] = {bitmap.get()};
    if (llama_dart_mtmd_tokenize(mtmd, chunks.get(), &text, bitmaps, 1) != 0) {
        std::fprintf(stderr, "ASR prompt tokenization failed\n");
        return {};
    }
    llama_memory_clear(llama_get_memory(context), true);
    llama_pos n_past = 0;
    if (llama_dart_mtmd_helper_eval_chunks(mtmd, context, chunks.get(), 0, 0, 512, true, &n_past) != 0) {
        std::fprintf(stderr, "ASR prompt evaluation failed\n");
        return {};
    }
    const llama_vocab *vocab = llama_model_get_vocab(model);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(
        llama_sampler_init_greedy(), llama_sampler_free);
    std::string transcript;
    for (int i = 0; i < 64; ++i) {
        llama_token token = llama_dart_sampler_sample(sampler.get(), context, -1);
        if (token == LLAMA_TOKEN_NULL || llama_vocab_is_eog(vocab, token)) {
            break;
        }
        char piece[256];
        const int32_t n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        if (n > 0) {
            transcript.append(piece, static_cast<size_t>(n));
        }
        if (llama_dart_decode(context, llama_batch_get_one(&token, 1)) != 0) {
            std::fprintf(stderr, "ASR decode failed\n");
            return {};
        }
    }
    llama_memory_clear(llama_get_memory(context), true);
    return transcript;
}

int main(int argc, char **argv) {
    const bool use_gpu = argc > 1 && std::strcmp(argv[argc - 1], "--gpu") == 0;
    const int value_argc = argc - (use_gpu ? 1 : 0);
    if (value_argc != 4) {
        std::fprintf(stderr, "usage: %s LFM3_AUDIO_MODEL MMPROJ OUTPUT_WAV [--gpu]\n", argv[0]);
        return 2;
    }
    if ((llama_dart_tts_supported_model_types() & lfm3_bit) == 0) {
        std::fprintf(stderr, "this build cannot drive LFM3-Audio\n");
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
    mtmd_params.cb_eval = llama_dart_tts_eval_callback;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
        mtmd_init_from_file(argv[2], model.get(), mtmd_params), mtmd_free);
    if (mtmd == nullptr) {
        std::fprintf(stderr, "failed to load mmproj\n");
        return 1;
    }

    // The Qwen3-only exports keep refusing the generator.
    llama_dart_tts_info legacy{};
    legacy.struct_size = sizeof(legacy);
    llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    if (llama_dart_tts_get_info(mtmd.get(), &legacy) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        legacy.model_type != LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN || legacy.sample_rate != 0 ||
        llama_dart_tts_init(context.get(), mtmd.get(), &status) != nullptr ||
        status != LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
        std::fprintf(stderr, "a Qwen3-only export accepted LFM3-Audio\n");
        return 1;
    }
    llama_dart_tts_info info{};
    info.struct_size = sizeof(info);
    if (llama_dart_tts_get_info_ext(mtmd.get(), qwen3_bit, &info) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        info.model_type != LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN) {
        std::fprintf(stderr, "a mask without LFM3_AUDIO accepted LFM3-Audio\n");
        return 1;
    }
    if (llama_dart_tts_get_info_ext(mtmd.get(), qwen3_bit | lfm3_bit, &info) != LLAMA_DART_TTS_STATUS_OK ||
        info.model_type != LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO || info.sample_rate <= 0 ||
        info.channels != 1 ||
        info.capabilities != (LLAMA_DART_TTS_CAPABILITY_LANGUAGE | LLAMA_DART_TTS_CAPABILITY_VOICE)) {
        std::fprintf(stderr, "invalid LFM3-Audio capability metadata\n");
        return 1;
    }
    std::unique_ptr<llama_dart_tts, decltype(&llama_dart_tts_free)> tts(
        llama_dart_tts_init_ext(context.get(), mtmd.get(), lfm3_bit, &status), llama_dart_tts_free);
    if (tts == nullptr || status != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to initialize LFM3-Audio TTS: %d\n", static_cast<int>(status));
        return 1;
    }

    llama_dart_tts_request request = llama_dart_tts_request_default();
    request.text = speech_text;
    request.text_length = std::strlen(speech_text);
    request.language = "en";
    request.seed = 1;

    const unsigned char speaker[] = {0};
    llama_dart_tts_request with_speaker = request;
    with_speaker.speaker_audio = speaker;
    with_speaker.speaker_audio_length = sizeof(speaker);
    if (llama_dart_tts_start(tts.get(), &with_speaker) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        llama_dart_tts_start_with_voice(tts.get(), &request, "child") !=
            LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT) {
        std::fprintf(stderr, "LFM3-Audio request validation failed\n");
        return 1;
    }

    llama_dart_tts_progress progress{};
    progress.struct_size = sizeof(progress);
    if (llama_dart_tts_start(tts.get(), &request) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start cancellation probe\n");
        return 1;
    }
    llama_dart_tts_cancel(tts.get());
    if (llama_dart_tts_step(tts.get(), &progress) != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED ||
        llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "cancel before the first step failed\n");
        return 1;
    }

    std::vector<float> first;
    llama_dart_tts_progress first_progress{};
    if (!synthesize(tts.get(), request, nullptr, info.sample_rate, &first, &first_progress)) {
        return 1;
    }
    if (first_progress.truncated || first_progress.frames_generated <= 0) {
        std::fprintf(stderr, "speech did not end on its own within %d frames\n", request.max_frames);
        return 1;
    }

    // A flag raised after a few frames cancels within the next step.
    std::atomic<int8_t> flag{0};
    static_assert(sizeof(std::atomic<int8_t>) == sizeof(int8_t), "the cancel flag must be a plain byte");
    if (llama_dart_tts_start(tts.get(), &request) != LLAMA_DART_TTS_STATUS_OK ||
        llama_dart_tts_set_cancel_flag(tts.get(), reinterpret_cast<const int8_t *>(&flag)) !=
            LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start the in-generation cancel probe\n");
        return 1;
    }
    int cancel_frames = -1;
    for (;;) {
        progress.struct_size = sizeof(progress);
        const llama_dart_tts_status step = llama_dart_tts_step(tts.get(), &progress);
        if (step == LLAMA_DART_TTS_STATUS_CANCELLED) {
            cancel_frames = progress.frames_generated;
            break;
        }
        if (step != LLAMA_DART_TTS_STATUS_OK || progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            std::fprintf(stderr, "the in-generation cancel did not land\n");
            return 1;
        }
        if (progress.frames_generated == 3) {
            flag.store(1);
        }
    }
    if (cancel_frames != 3 || llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "the in-generation cancel landed after %d frames\n", cancel_frames);
        return 1;
    }

    std::vector<float> repeat;
    llama_dart_tts_progress repeat_progress{};
    request.sequence_id = 1;
    if (!synthesize(tts.get(), request, "FEMALE", info.sample_rate, &repeat, &repeat_progress)) {
        return 1;
    }
    if (!same_pcm(first, repeat)) {
        std::fprintf(stderr, "the same seed and default voice did not repeat the PCM after cancel and reuse\n");
        return 1;
    }

    std::vector<float> male;
    llama_dart_tts_progress male_progress{};
    if (!synthesize(tts.get(), request, "male", info.sample_rate, &male, &male_progress)) {
        return 1;
    }
    if (same_pcm(first, male)) {
        std::fprintf(stderr, "the voice did not reach the chat template\n");
        return 1;
    }

    std::vector<float> german;
    llama_dart_tts_progress german_progress{};
    llama_dart_tts_request german_request = request;
    german_request.language = "de";
    if (!synthesize(tts.get(), german_request, nullptr, info.sample_rate, &german, &german_progress)) {
        return 1;
    }
    if (same_pcm(first, german)) {
        std::fprintf(stderr, "the language did not reach the chat template\n");
        return 1;
    }

    std::vector<float> reseeded;
    llama_dart_tts_progress reseeded_progress{};
    llama_dart_tts_request seed_request = request;
    seed_request.seed = 2;
    if (!synthesize(tts.get(), seed_request, nullptr, info.sample_rate, &reseeded, &reseeded_progress)) {
        return 1;
    }
    if (same_pcm(first, reseeded)) {
        std::fprintf(stderr, "the seed did not reach the code sampler\n");
        return 1;
    }

    llama_dart_tts_request short_request = request;
    short_request.max_frames = 4;
    std::vector<float> truncated;
    llama_dart_tts_progress truncated_progress{};
    if (!synthesize(tts.get(), short_request, nullptr, info.sample_rate, &truncated, &truncated_progress) ||
        !truncated_progress.truncated || truncated_progress.frames_generated != 4) {
        std::fprintf(stderr, "max_frames did not truncate the speech\n");
        return 1;
    }
    if (llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "final reset failed\n");
        return 1;
    }

    const std::vector<unsigned char> wav = wav16(first, info.sample_rate);
    std::string transcript = transcribe(context.get(), mtmd.get(), model.get(), wav);
    std::string lowered = transcript;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lowered.find("hello") == std::string::npos || lowered.find("today") == std::string::npos) {
        std::fprintf(stderr, "the transcript of the speech is \"%s\"\n", transcript.c_str());
        return 1;
    }
    std::ofstream out(argv[3], std::ios::binary);
    out.write(reinterpret_cast<const char *>(wav.data()), static_cast<std::streamsize>(wav.size()));
    if (!out) {
        std::fprintf(stderr, "failed to write WAV output\n");
        return 1;
    }
    std::printf("PASS backend=%s sample_rate=%d frames=%d samples=%zu rms=%.6f male_frames=%d "
                "cancel_frames=%d transcript=\"%s\"\n",
                use_gpu ? "gpu" : "cpu", info.sample_rate, first_progress.frames_generated, first.size(),
                finite_rms(first), male_progress.frames_generated, cancel_frames, transcript.c_str());
    return 0;
}
