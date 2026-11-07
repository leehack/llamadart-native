// Local check of the opt-in Chatterbox and Pocket TTS generators and their
// predefined voices against real weights; see docs/tts_voices.md.
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
#include <iterator>
#include <memory>
#include <string>
#include <vector>

static const char *const speech_text = "Hello from Llama Dart. How are you today?";
static const char *const french_text = "Bonjour, comment allez-vous aujourd'hui ?";

static constexpr uint32_t qwen3_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
static constexpr uint32_t chatterbox_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
static constexpr uint32_t pocket_bit = 1u << LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS;
static constexpr uint32_t all_bits = qwen3_bit | (1u << LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO) |
                                     (1u << LLAMA_DART_TTS_MODEL_TYPE_SOPRANO) | chatterbox_bit |
                                     pocket_bit;

struct llama_backend_guard {
    llama_backend_guard() { llama_backend_init(); }
    ~llama_backend_guard() { llama_backend_free(); }
};

using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using tts_ptr = std::unique_ptr<llama_dart_tts, decltype(&llama_dart_tts_free)>;

static bool fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    return false;
}

static int die(const char *message) {
    fail(message);
    return 1;
}

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

static bool write_file(const std::string &path, const std::vector<unsigned char> &bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

static double finite_rms(const std::vector<float> &pcm) {
    double energy = 0.0;
    for (float sample : pcm) {
        if (!std::isfinite(sample)) {
            return -1.0;
        }
        energy += static_cast<double>(sample) * sample;
    }
    return pcm.empty() ? 0.0 : std::sqrt(energy / pcm.size());
}

static bool same_pcm(const std::vector<float> &a, const std::vector<float> &b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static bool sequence_empty(llama_context *context, llama_seq_id sequence) {
    return llama_memory_seq_pos_max(llama_get_memory(context), sequence) < 0;
}

static context_ptr make_context(llama_model *model, uint32_t n_seq_max) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 8192;
    params.n_batch = 512;
    params.n_ubatch = 512;
    params.n_seq_max = n_seq_max;
    // As upstream llama-tts does for Chatterbox Multilingual's paired sequence.
    params.kv_unified = true;
    params.embeddings = true;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

struct synthesis {
    std::vector<float> pcm;
    llama_dart_tts_progress progress{};
};

static bool synthesize(llama_dart_tts *tts, const llama_dart_tts_request &request, const char *voice,
                       int32_t sample_rate, synthesis *out) {
    if (llama_dart_tts_start_with_voice(tts, &request, voice) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start TTS: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    do {
        out->progress.struct_size = sizeof(out->progress);
        if (llama_dart_tts_step(tts, &out->progress) != LLAMA_DART_TTS_STATUS_OK) {
            std::fprintf(stderr, "TTS step failed after %d frames: %s\n", out->progress.frames_generated,
                         llama_dart_tts_last_error(tts));
            return false;
        }
    } while (out->progress.state != LLAMA_DART_TTS_STATE_COMPLETED);
    llama_dart_tts_output_info output{};
    output.struct_size = sizeof(output);
    if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK ||
        output.sample_rate != sample_rate || output.channels != 1 || output.sample_count <= 0) {
        return die("invalid TTS output metadata");
    }
    out->pcm.resize(static_cast<size_t>(output.sample_count));
    size_t read = 0;
    if (llama_dart_tts_read_pcm(tts, 0, out->pcm.data(), out->pcm.size(), &read) != LLAMA_DART_TTS_STATUS_OK ||
        read != out->pcm.size()) {
        return fail("failed to read PCM");
    }
    const double rms = finite_rms(out->pcm);
    if (rms < 1e-4) {
        std::fprintf(stderr, "PCM is non-finite or silent (rms %.6f)\n", rms);
        return false;
    }
    return true;
}

static bool list_voices(mtmd_context *mtmd, uint32_t bit, std::vector<std::string> *voices) {
    int32_t count = -1;
    if (llama_dart_tts_get_voice_count(mtmd, qwen3_bit, &count) != LLAMA_DART_TTS_STATUS_UNSUPPORTED || count != 0) {
        return fail("a mask without the generator listed its voices");
    }
    if (llama_dart_tts_get_voice_count(mtmd, bit, &count) != LLAMA_DART_TTS_STATUS_OK || count <= 0) {
        return fail("the projector lists no predefined voice");
    }
#if LLAMADART_MTMD_HAS_GEN_VOICES
    if (count != mtmd_gen_audio_get_info(mtmd).n_voices) {
        return fail("the wrapper's voice count differs from upstream's");
    }
#endif
    for (int32_t i = 0; i < count; ++i) {
        size_t length = 0;
        if (llama_dart_tts_get_voice_name(mtmd, bit, i, nullptr, 0, &length) != LLAMA_DART_TTS_STATUS_OK ||
            length == 0) {
            return fail("voice name length query failed");
        }
        std::string name(length + 1, '\0');
        size_t short_length = 0;
        if (llama_dart_tts_get_voice_name(mtmd, bit, i, name.data(), length, &short_length) !=
                LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT ||
            short_length != length) {
            return fail("a buffer without room for the NUL was accepted");
        }
        if (llama_dart_tts_get_voice_name(mtmd, bit, i, name.data(), name.size(), &length) !=
                LLAMA_DART_TTS_STATUS_OK ||
            name[length] != '\0') {
            return fail("voice name copy failed");
        }
        name.resize(length);
        voices->push_back(name);
    }
    size_t length = 0;
    if (llama_dart_tts_get_voice_name(mtmd, bit, count, nullptr, 0, &length) !=
            LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT ||
        llama_dart_tts_get_voice_name(mtmd, bit, -1, nullptr, 0, &length) !=
            LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT) {
        return fail("an out-of-range voice index was accepted");
    }
    return true;
}

static std::string upper(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return text;
}

static bool cancel_in_generation(llama_dart_tts *tts, const llama_dart_tts_request &request,
                                 int *cancel_frames) {
    std::atomic<int8_t> flag{0};
    static_assert(sizeof(std::atomic<int8_t>) == sizeof(int8_t), "the cancel flag must be a plain byte");
    if (llama_dart_tts_start(tts, &request) != LLAMA_DART_TTS_STATUS_OK ||
        llama_dart_tts_set_cancel_flag(tts, reinterpret_cast<const int8_t *>(&flag)) !=
            LLAMA_DART_TTS_STATUS_OK) {
        return fail("failed to start the in-generation cancel probe");
    }
    llama_dart_tts_progress progress{};
    for (;;) {
        progress.struct_size = sizeof(progress);
        const llama_dart_tts_status step = llama_dart_tts_step(tts, &progress);
        if (step == LLAMA_DART_TTS_STATUS_CANCELLED) {
            *cancel_frames = progress.frames_generated;
            return true;
        }
        if (step != LLAMA_DART_TTS_STATUS_OK || progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            return fail("the in-generation cancel did not land");
        }
        if (progress.frames_generated == 3) {
            flag.store(1);
        }
    }
}

int main(int argc, char **argv) {
    bool use_gpu = false;
    const char *speaker_path = nullptr;
    std::vector<const char *> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--gpu") == 0) {
            use_gpu = true;
        } else if (std::strcmp(argv[i], "--speaker") == 0 && i + 1 < argc) {
            speaker_path = argv[++i];
        } else {
            positional.push_back(argv[i]);
        }
    }
    if (positional.size() != 3) {
        std::fprintf(stderr, "usage: %s MODEL MMPROJ OUTPUT_PREFIX [--speaker WAV] [--gpu]\n", argv[0]);
        return 2;
    }
    const std::string output_prefix = positional[2];
    std::vector<unsigned char> speaker_wav;
    if (speaker_path != nullptr) {
        std::ifstream in(speaker_path, std::ios::binary);
        speaker_wav.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        if (speaker_wav.empty()) {
            return fail("failed to read the speaker reference");
        }
    }

    const llama_backend_guard backend;
    llama_dart_set_log_level(4);
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = use_gpu ? 99 : 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(positional[0], model_params), llama_model_free);
    if (model == nullptr) {
        return die("failed to load model");
    }
    context_ptr context = make_context(model.get(), 4);
    if (context == nullptr) {
        return die("failed to create context");
    }
    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = use_gpu;
    mtmd_params.cb_eval = llama_dart_tts_eval_callback;
    std::unique_ptr<mtmd_context, decltype(&mtmd_free)> mtmd(
        mtmd_init_from_file(positional[1], model.get(), mtmd_params), mtmd_free);
    if (mtmd == nullptr) {
        return die("failed to load mmproj");
    }

    llama_dart_tts_info info{};
    info.struct_size = sizeof(info);
    if (llama_dart_tts_get_info_ext(mtmd.get(), all_bits, &info) != LLAMA_DART_TTS_STATUS_OK ||
        (info.model_type != LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX &&
         info.model_type != LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS)) {
        return die("the projector is neither Chatterbox nor Pocket TTS, or this build cannot drive it");
    }
    const bool chatterbox = info.model_type == LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX;
    const uint32_t bit = chatterbox ? chatterbox_bit : pocket_bit;
    const bool takes_language = (info.capabilities & LLAMA_DART_TTS_CAPABILITY_LANGUAGE) != 0;
    const bool takes_speaker = (info.capabilities & LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE) != 0;
    if ((info.capabilities & LLAMA_DART_TTS_CAPABILITY_VOICE) == 0 || info.channels != 1 ||
        info.sample_rate <= 0 || (!chatterbox && takes_language)) {
        return die("invalid capability metadata");
    }

    // Callers that did not opt in keep seeing UNKNOWN.
    llama_dart_tts_info legacy{};
    legacy.struct_size = sizeof(legacy);
    llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    if (llama_dart_tts_get_info(mtmd.get(), &legacy) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        legacy.model_type != LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN ||
        llama_dart_tts_init(context.get(), mtmd.get(), &status) != nullptr ||
        status != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        llama_dart_tts_get_info_ext(mtmd.get(), all_bits & ~bit, &legacy) != LLAMA_DART_TTS_STATUS_UNSUPPORTED ||
        llama_dart_tts_init_ext(context.get(), mtmd.get(), all_bits & ~bit, &status) != nullptr ||
        status != LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
        return die("an export accepted a generator outside its mask");
    }

    std::vector<std::string> voices;
    if (!list_voices(mtmd.get(), bit, &voices)) {
        return 1;
    }

    if (takes_language) {
        // The unconditional sequence needs a second sequence in the context.
        context_ptr single = make_context(model.get(), 1);
        status = LLAMA_DART_TTS_STATUS_OK;
        if (single == nullptr || llama_dart_tts_init_ext(single.get(), mtmd.get(), bit, &status) != nullptr ||
            status != LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
            return die("Chatterbox Multilingual accepted a context without a paired sequence");
        }
    }

    tts_ptr tts(llama_dart_tts_init_ext(context.get(), mtmd.get(), bit, &status), llama_dart_tts_free);
    if (tts == nullptr || status != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to initialize TTS: %d\n", static_cast<int>(status));
        return 1;
    }

    llama_dart_tts_request request = llama_dart_tts_request_default();
    request.text = speech_text;
    request.text_length = std::strlen(speech_text);
    request.language = takes_language ? "en" : nullptr;
    request.seed = 1;

    const unsigned char junk[] = {0};
    llama_dart_tts_request with_speaker = request;
    with_speaker.speaker_audio = junk;
    with_speaker.speaker_audio_length = sizeof(junk);
    llama_dart_tts_request out_of_range = request;
    out_of_range.sequence_id = 2;
    llama_dart_tts_request with_language = request;
    with_language.language = "fr";
    if (llama_dart_tts_start_with_voice(tts.get(), &request, "nobody") != LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT ||
        llama_dart_tts_start_with_voice(tts.get(), &request, "") != LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT ||
        llama_dart_tts_start_with_voice(tts.get(), &with_speaker, voices[0].c_str()) !=
            (takes_speaker ? LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT : LLAMA_DART_TTS_STATUS_UNSUPPORTED) ||
        (!takes_speaker && llama_dart_tts_start(tts.get(), &with_speaker) != LLAMA_DART_TTS_STATUS_UNSUPPORTED) ||
        (!takes_language &&
         llama_dart_tts_start(tts.get(), &with_language) != LLAMA_DART_TTS_STATUS_UNSUPPORTED) ||
        (takes_language &&
         llama_dart_tts_start(tts.get(), &out_of_range) != LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT)) {
        return die("request validation failed");
    }

    llama_dart_tts_progress progress{};
    progress.struct_size = sizeof(progress);
    if (llama_dart_tts_start(tts.get(), &request) != LLAMA_DART_TTS_STATUS_OK) {
        return die("failed to start the cancellation probe");
    }
    llama_dart_tts_cancel(tts.get());
    if (llama_dart_tts_step(tts.get(), &progress) != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED ||
        llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        return die("cancel before the first step failed");
    }

    synthesis first;
    if (!synthesize(tts.get(), request, nullptr, info.sample_rate, &first)) {
        return 1;
    }
    if (first.progress.truncated || first.progress.frames_generated <= 0) {
        std::fprintf(stderr, "speech did not end on its own within %d frames\n", request.max_frames);
        return 1;
    }
    // The task owns both sequences only while it runs.
    if (!sequence_empty(context.get(), 0) || !sequence_empty(context.get(), 2)) {
        return die("a completed task left its sequences in the context");
    }

    int cancel_frames = -1;
    if (!cancel_in_generation(tts.get(), request, &cancel_frames)) {
        return 1;
    }
    if (cancel_frames != 3 || !sequence_empty(context.get(), 0) || !sequence_empty(context.get(), 2) ||
        llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "the in-generation cancel landed after %d frames or left its sequences\n",
                     cancel_frames);
        return 1;
    }

    // NULL is the first listed voice, and a voice matches in any case.
    synthesis named;
    llama_dart_tts_request reuse = request;
    reuse.sequence_id = 1;
    const std::string first_voice = upper(voices[0]);
    if (!synthesize(tts.get(), reuse, first_voice.c_str(), info.sample_rate, &named)) {
        return 1;
    }
    if (!same_pcm(first.pcm, named.pcm)) {
        return die("the default voice is not the first listed voice, or reuse changed the PCM");
    }
    if (!sequence_empty(context.get(), 1) || !sequence_empty(context.get(), 3)) {
        return die("a completed task left its sequences in the context");
    }

    synthesis other;
    if (voices.size() > 1) {
        if (!synthesize(tts.get(), request, voices.back().c_str(), info.sample_rate, &other)) {
            return 1;
        }
        if (same_pcm(first.pcm, other.pcm)) {
            return die("the voice did not reach the generator");
        }
    }

    synthesis reseeded;
    llama_dart_tts_request seed_request = request;
    seed_request.seed = request.seed + 1;
    if (!synthesize(tts.get(), seed_request, nullptr, info.sample_rate, &reseeded)) {
        return 1;
    }
    if (same_pcm(first.pcm, reseeded.pcm)) {
        return die("the seed did not reach the generator");
    }

    synthesis cloned;
    if (takes_speaker && !speaker_wav.empty()) {
        llama_dart_tts_request clone = request;
        clone.speaker_audio = speaker_wav.data();
        clone.speaker_audio_length = speaker_wav.size();
        if (!synthesize(tts.get(), clone, nullptr, info.sample_rate, &cloned)) {
            return 1;
        }
        if (cloned.progress.truncated || same_pcm(first.pcm, cloned.pcm)) {
            return die("the speaker reference did not condition the speech, or it did not end");
        }
    } else if (!speaker_wav.empty()) {
        llama_dart_tts_request clone = request;
        clone.speaker_audio = speaker_wav.data();
        clone.speaker_audio_length = speaker_wav.size();
        if (llama_dart_tts_start(tts.get(), &clone) != LLAMA_DART_TTS_STATUS_UNSUPPORTED) {
            return die("a projector without a speaker encoder accepted a speaker reference");
        }
    }

    synthesis french;
    if (takes_language) {
        llama_dart_tts_request french_request = request;
        french_request.text = french_text;
        french_request.text_length = std::strlen(french_text);
        french_request.language = "fr";
        if (!synthesize(tts.get(), french_request, nullptr, info.sample_rate, &french)) {
            return 1;
        }
        if (french.progress.truncated) {
            return die("the French speech did not end");
        }
    }

    llama_dart_tts_request short_request = request;
    short_request.max_frames = 4;
    synthesis truncated;
    if (!synthesize(tts.get(), short_request, nullptr, info.sample_rate, &truncated) ||
        !truncated.progress.truncated || truncated.progress.frames_generated != 4) {
        return die("max_frames did not truncate the speech");
    }
    if (llama_dart_tts_reset(tts.get()) != LLAMA_DART_TTS_STATUS_OK) {
        return die("final reset failed");
    }

    if (!write_file(output_prefix + "-default.wav", wav16(first.pcm, info.sample_rate)) ||
        (!other.pcm.empty() &&
         !write_file(output_prefix + "-" + voices.back() + ".wav", wav16(other.pcm, info.sample_rate))) ||
        (!cloned.pcm.empty() && !write_file(output_prefix + "-cloned.wav", wav16(cloned.pcm, info.sample_rate))) ||
        (!french.pcm.empty() && !write_file(output_prefix + "-fr.wav", wav16(french.pcm, info.sample_rate)))) {
        return die("failed to write WAV output");
    }
    std::printf("PASS type=%s backend=%s capabilities=0x%x voices=%zu first_voice=%s last_voice=%s "
                "sample_rate=%d frames=%d samples=%zu rms=%.6f last_voice_frames=%d cloned_frames=%d "
                "cancel_frames=%d\n",
                chatterbox ? (takes_language ? "chatterbox-multilingual" : "chatterbox-turbo") : "pocket-tts",
                use_gpu ? "gpu" : "cpu", info.capabilities, voices.size(), voices.front().c_str(),
                voices.back().c_str(), info.sample_rate, first.progress.frames_generated, first.pcm.size(),
                finite_rms(first.pcm), other.progress.frames_generated, cloned.progress.frames_generated,
                cancel_frames);
    return 0;
}
