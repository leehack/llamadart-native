#include "llama_dart_image_token_limit_internal.h"
#include "clip-impl.h"
#include "gguf.h"

#include <cstring>
#include <memory>
#include <stdexcept>

int32_t llama_dart_image_token_limit_supported(const char *mmproj_fname,
                                             int32_t max_tokens) {
  const gguf_init_params params{true, nullptr};
  std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(
      gguf_init_from_file(mmproj_fname, params), gguf_free);
  if (!metadata) {
    throw std::runtime_error("could not read projector GGUF metadata");
  }
  auto key = gguf_find_key(metadata.get(), "clip.projector_type");
  if (key >= 0 && gguf_get_kv_type(metadata.get(), key) != GGUF_TYPE_STRING) {
    return 0;
  }
  if (key < 0 || std::strlen(gguf_get_val_str(metadata.get(), key)) == 0) {
    key = gguf_find_key(metadata.get(), "clip.vision.projector_type");
  }
  if (key < 0 || gguf_get_kv_type(metadata.get(), key) != GGUF_TYPE_STRING) {
    return 0;
  }
  const char *type = gguf_get_val_str(metadata.get(), key);
  if (clip_projector_type_from_string(type) == PROJECTOR_TYPE_UNKNOWN) {
    return 0;
  }
  // v0.6.0 applies custom_image_max_tokens for these dynamic projectors
  // through set_limit_image_tokens(70, 1120). Never raise their metadata
  // budget or set a maximum below the default minimum-pixel threshold.
  return (std::strcmp(type, "gemma4v") == 0 ||
          std::strcmp(type, "gemma4uv") == 0) &&
                 max_tokens >= 70 && max_tokens <= 1120
             ? 1
             : 0;
}
