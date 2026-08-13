#include "llama_dart_wrapper.h"

#include <assert.h>
#include <string.h>

int main(void) {
    assert(llama_dart_tts_api_version() == LLAMA_DART_TTS_API_VERSION);

    struct llama_dart_tts_request request = llama_dart_tts_request_default();
    assert(request.struct_size == sizeof(request));
    assert(request.sequence_id == 0);
    assert(request.prompt_batch_size == 512);
    assert(request.max_frames == 512);
    assert(request.top_k == 40);
    assert(request.top_p == 0.95f);
    assert(request.min_p == 0.0f);
    assert(request.temperature == 0.8f);
    assert(request.seed == LLAMA_DEFAULT_SEED);

    struct llama_dart_tts_info info = {0};
    info.struct_size = sizeof(info);
    assert(llama_dart_tts_get_info(NULL, &info) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_tts_get_info(NULL, NULL) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    info.struct_size = sizeof(info) - 1;
    assert(llama_dart_tts_get_info(NULL, &info) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);

    enum llama_dart_tts_status status = LLAMA_DART_TTS_STATUS_OK;
    assert(llama_dart_tts_init(NULL, NULL, &status) == NULL);
    assert(status == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);

    struct llama_dart_tts_progress progress = {0};
    progress.struct_size = sizeof(progress);
    assert(llama_dart_tts_start(NULL, &request) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_tts_step(NULL, &progress) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    progress.struct_size = sizeof(progress) - 1;
    assert(llama_dart_tts_step(NULL, &progress) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    assert(llama_dart_tts_reset(NULL) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);

    struct llama_dart_tts_output_info output = {0};
    output.struct_size = sizeof(output);
    assert(llama_dart_tts_get_output_info(NULL, &output) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    output.struct_size = sizeof(output) - 1;
    assert(llama_dart_tts_get_output_info(NULL, &output) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);

    size_t count = 0;
    assert(llama_dart_tts_read_pcm(NULL, 0, NULL, 0, &count) == LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT);
    assert(strcmp(llama_dart_tts_last_error(NULL), "invalid TTS handle") == 0);

    llama_dart_tts_cancel(NULL);
    llama_dart_tts_free(NULL);
    return 0;
}
