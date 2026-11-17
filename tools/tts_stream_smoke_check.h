// Streamed-PCM checks shared by the local TTS smokes; see docs/tts_streaming.md.
#pragma once

#include "llama_dart_wrapper.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace tts_stream_smoke {

struct expectation {
    // The frame count after which the first samples appear and their number, 0
    // for no expectation.
    int first_frames = 0;
    int64_t first_samples = 0;
    // The task must grow its PCM in at least this many steps.
    int min_growth_steps = 2;
};

struct streamed_run {
    std::vector<float> pcm;
    int growth_steps = 0;
    int64_t first_samples = 0;
    int first_frames = 0;
};

inline bool fail(const char *message) {
    std::fprintf(stderr, "%s\n", message);
    return false;
}

// Reads the whole PCM the task exposes right now.
inline bool read_all(const llama_dart_tts *tts, int64_t expected_count, std::vector<float> *pcm) {
    size_t remaining = 0;
    if (llama_dart_tts_read_pcm(tts, 0, nullptr, 0, &remaining) != LLAMA_DART_TTS_STATUS_OK ||
        static_cast<int64_t>(remaining) != expected_count) {
        return fail("the PCM length query disagrees with the output info");
    }
    pcm->resize(remaining);
    size_t read = 0;
    if (llama_dart_tts_read_pcm(tts, 0, pcm->data(), pcm->size(), &read) != LLAMA_DART_TTS_STATUS_OK ||
        read != pcm->size()) {
        return fail("failed to read the PCM");
    }
    return true;
}

// Both reads must refuse a task that is not readable.
inline bool reads_refused(const llama_dart_tts *tts) {
    llama_dart_tts_output_info output{};
    output.struct_size = sizeof(output);
    size_t count = 0;
    float sample = 0.0f;
    return llama_dart_tts_get_output_info(tts, &output) == LLAMA_DART_TTS_STATUS_INVALID_STATE &&
           llama_dart_tts_read_pcm(tts, 0, nullptr, 0, &count) == LLAMA_DART_TTS_STATUS_INVALID_STATE &&
           llama_dart_tts_read_pcm(tts, 0, &sample, 1, &count) == LLAMA_DART_TTS_STATUS_INVALID_STATE;
}

// Runs a streamed task to COMPLETED, reading the exposed PCM after every step.
inline bool run_streamed(llama_dart_tts *tts, const llama_dart_tts_request &request, const char *voice,
                         int32_t sample_rate, const expectation &expected, streamed_run *run) {
    if (llama_dart_tts_start_streaming(tts, &request, voice) != LLAMA_DART_TTS_STATUS_OK) {
        std::fprintf(stderr, "failed to start the streamed task: %s\n", llama_dart_tts_last_error(tts));
        return false;
    }
    llama_dart_tts_progress progress{};
    int64_t previous = 0;
    std::vector<float> first_chunk;
    for (;;) {
        progress.struct_size = sizeof(progress);
        const llama_dart_tts_status status = llama_dart_tts_step(tts, &progress);
        if (status != LLAMA_DART_TTS_STATUS_OK) {
            std::fprintf(stderr, "streamed step failed: %s\n", llama_dart_tts_last_error(tts));
            return false;
        }
        llama_dart_tts_output_info output{};
        output.struct_size = sizeof(output);
        if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK ||
            output.sample_rate != sample_rate || output.channels != 1 || output.sample_count < previous) {
            return fail("the streamed output info is invalid or shrank");
        }
        std::vector<float> current;
        if (!read_all(tts, output.sample_count, &current)) {
            return false;
        }
        if (!first_chunk.empty() &&
            std::memcmp(first_chunk.data(), current.data(), first_chunk.size() * sizeof(float)) != 0) {
            return fail("an already readable sample changed");
        }
        if (output.sample_count > previous) {
            ++run->growth_steps;
            if (previous == 0) {
                run->first_samples = output.sample_count;
                run->first_frames = progress.frames_generated;
                first_chunk = current;
            }
        }
        previous = output.sample_count;
        if (progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            run->pcm = current;
            break;
        }
    }
    if (run->growth_steps < expected.min_growth_steps) {
        std::fprintf(stderr, "the PCM grew in %d step(s), expected at least %d\n", run->growth_steps,
                     expected.min_growth_steps);
        return false;
    }
    if (expected.first_samples > 0 && run->first_samples != expected.first_samples) {
        std::fprintf(stderr, "the first readable window held %lld samples, expected %lld\n",
                     static_cast<long long>(run->first_samples), static_cast<long long>(expected.first_samples));
        return false;
    }
    if (expected.first_frames > 0 && run->first_frames != expected.first_frames) {
        std::fprintf(stderr, "the first samples appeared after %d frames, expected %d\n", run->first_frames,
                     expected.first_frames);
        return false;
    }
    return true;
}

// A task started without streaming keeps its PCM back until it completes.
inline bool plain_task_hides_pcm(llama_dart_tts *tts, const llama_dart_tts_request &request, const char *voice,
                                 int frames) {
    if (llama_dart_tts_start_with_voice(tts, &request, voice) != LLAMA_DART_TTS_STATUS_OK) {
        return fail("failed to start the plain task");
    }
    llama_dart_tts_progress progress{};
    do {
        progress.struct_size = sizeof(progress);
        if (llama_dart_tts_step(tts, &progress) != LLAMA_DART_TTS_STATUS_OK) {
            return fail("plain step failed");
        }
        if (progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            return fail("the plain task completed before it was probed");
        }
        if (!reads_refused(tts)) {
            return fail("a plain task exposed PCM before COMPLETED");
        }
    } while (progress.frames_generated < frames);
    return llama_dart_tts_reset(tts) == LLAMA_DART_TTS_STATUS_OK;
}

// A cancel after the first readable window ends the task and withdraws its PCM.
inline bool streamed_cancel(llama_dart_tts *tts, const llama_dart_tts_request &request, const char *voice) {
    if (llama_dart_tts_start_streaming(tts, &request, voice) != LLAMA_DART_TTS_STATUS_OK) {
        return fail("failed to start the streamed cancel probe");
    }
    llama_dart_tts_progress progress{};
    for (int64_t count = 0; count == 0;) {
        progress.struct_size = sizeof(progress);
        if (llama_dart_tts_step(tts, &progress) != LLAMA_DART_TTS_STATUS_OK ||
            progress.state == LLAMA_DART_TTS_STATE_COMPLETED) {
            return fail("the streamed task ended before its first window");
        }
        llama_dart_tts_output_info output{};
        output.struct_size = sizeof(output);
        if (llama_dart_tts_get_output_info(tts, &output) != LLAMA_DART_TTS_STATUS_OK) {
            return fail("a streamed task refused its output info while generating");
        }
        count = output.sample_count;
    }
    llama_dart_tts_cancel(tts);
    progress.struct_size = sizeof(progress);
    if (llama_dart_tts_step(tts, &progress) != LLAMA_DART_TTS_STATUS_CANCELLED ||
        progress.state != LLAMA_DART_TTS_STATE_CANCELLED) {
        return fail("the streamed cancel did not land");
    }
    if (!reads_refused(tts)) {
        return fail("a cancelled streamed task still exposed PCM");
    }
    if (llama_dart_tts_reset(tts) != LLAMA_DART_TTS_STATUS_OK || !reads_refused(tts)) {
        return fail("a reset streamed task still exposed PCM");
    }
    return true;
}

} // namespace tts_stream_smoke
