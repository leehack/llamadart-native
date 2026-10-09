#pragma once
#include <cstdint>

// Reads upstream projector metadata; the public wrapper guards the call and
// translates exceptions into llama_dart_last_error.
int32_t llama_dart_image_token_limit_supported(const char *path, int32_t max_tokens);
