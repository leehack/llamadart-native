#include "llama_dart_wrapper.h"
#include "gguf.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>
#include "mtmd.h"

#ifdef NDEBUG
#error "Wrapper contract tests require active assertions"
#endif

static void real_image_limit(const char *model_path, const char *projector_path,
                             int limit, bool encode) {
  assert(llama_dart_mtmd_supports_image_token_limit(projector_path, limit) == 1);
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = 0;
  model_params.vocab_only = true;
  llama_model *model = llama_dart_model_load_from_file(model_path, model_params);
  assert(model != nullptr);
  std::vector<unsigned char> pixels(1600 * 1600 * 3, 127);
  mtmd_bitmap *bitmap = mtmd_bitmap_init(1600, 1600, pixels.data());
  const mtmd_bitmap *bitmaps[] = {bitmap};
  size_t counts[2] = {};
  for (int run = 0; run < 2; ++run) {
    auto params = mtmd_context_params_default();
    params.use_gpu = false;
    params.warmup = false;
    params.n_threads = 4;
    if (run == 1) params.image_max_tokens = limit;
    mtmd_context *projector = llama_dart_mtmd_init_from_file(projector_path, model, &params);
    assert(projector != nullptr);
    auto *chunks = mtmd_input_chunks_init();
    const char *marker = mtmd_get_marker(projector);
    const mtmd_input_text text{marker, std::strlen(marker), true, true};
    assert(llama_dart_mtmd_tokenize(projector, chunks, &text, bitmaps, 1) == 0);
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
      const auto *chunk = mtmd_input_chunks_get(chunks, i);
      if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
        counts[run] += mtmd_input_chunk_get_n_tokens(chunk);
        if (encode && run == 1) {
          assert(llama_dart_mtmd_encode_chunk(projector, chunk) == 0);
        }
      }
    }
    mtmd_input_chunks_free(chunks);
    llama_dart_exit_free(projector);
  }
  std::printf("real image tokens: default=%zu capped=%zu\n", counts[0], counts[1]);
  assert(counts[0] > static_cast<size_t>(limit) && counts[1] > 0 &&
         counts[1] <= static_cast<size_t>(limit));
  mtmd_bitmap_free(bitmap);
  llama_dart_exit_free(model);
}

int main(int argc, char **argv) {
  if (argc == 3 || argc == 4) {
    const int limit = argc == 4 ? std::atoi(argv[3]) : 512;
    real_image_limit(argv[1], argv[2], limit, argc == 4);
    return 0;
  }
  const char *path = "image_token_limit_metadata.gguf";
  const auto check = [path](const char *key, const char *type, int limit, int expected) {
    gguf_context *metadata = gguf_init_empty();
    if (type != nullptr) gguf_set_val_str(metadata, key, type);
    assert(gguf_write_to_file(metadata, path, true));
    gguf_free(metadata);
    assert(llama_dart_mtmd_supports_image_token_limit(path, limit) == expected);
  };
  for (const char *type : {"gemma4v", "gemma4uv"}) {
    for (int limit : {70, 256, 512, 1120}) {
      check("clip.projector_type", type, limit, 1);
      check("clip.vision.projector_type", type, limit, 1);
    }
    check("clip.projector_type", type, 69, 0);
    check("clip.projector_type", type, 1121, 0);
  }
  for (const char *type : {"gemma3", "mlp", "gemma4a", "unknown"}) {
    check("clip.projector_type", type, 512, 0);
  }
  check("clip.projector_type", nullptr, 512, 0);
  gguf_context *mixed = gguf_init_empty();
  gguf_set_val_str(mixed, "clip.projector_type", "gemma3");
  gguf_set_val_str(mixed, "clip.vision.projector_type", "gemma4v");
  assert(gguf_write_to_file(mixed, path, true));
  assert(llama_dart_mtmd_supports_image_token_limit(path, 512) == 0);
  gguf_set_val_str(mixed, "clip.projector_type", "");
  assert(gguf_write_to_file(mixed, path, true));
  assert(llama_dart_mtmd_supports_image_token_limit(path, 512) == 1);
  gguf_free(mixed);
  gguf_context *metadata = gguf_init_empty();
  gguf_set_val_u32(metadata, "clip.projector_type", 7);
  assert(gguf_write_to_file(metadata, path, true));
  gguf_free(metadata);
  assert(llama_dart_mtmd_supports_image_token_limit(path, 512) == 0);
  assert(llama_dart_mtmd_supports_image_token_limit(path, 0) == -1);
  assert(std::strstr(llama_dart_last_error(), "positive limit") != nullptr);
  assert(llama_dart_mtmd_supports_image_token_limit(nullptr, 512) == -1);
  std::remove(path);
  assert(llama_dart_mtmd_supports_image_token_limit(path, 512) == -1);
  assert(std::strstr(llama_dart_last_error(), "GGUF metadata") != nullptr);
  // A later successful probe clears the caught-error slot.
  check("clip.projector_type", "gemma4v", 512, 1);
  assert(llama_dart_last_error() == nullptr || llama_dart_last_error()[0] == 0);
  std::remove(path);
}
