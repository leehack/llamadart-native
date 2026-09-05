#include "llama_dart_mtmd_compat.h"

#include <cstdio>

int main() {
  // A real one-pixel PPM exercises buffer forwarding and non-placeholder
  // decoding without a model. TTS rejects this non-audio result at its call site.
  const unsigned char image[] = "P6\n1 1\n255\n\xff\x00\x00";
  auto result = llama_dart_bitmap_from_buffer(nullptr, image, sizeof(image) - 1);
  if (!result.bitmap || result.video_ctx || mtmd_bitmap_is_audio(result.bitmap) ||
      mtmd_bitmap_get_nx(result.bitmap) != 1 ||
      mtmd_bitmap_get_ny(result.bitmap) != 1 ||
      !mtmd_bitmap_get_data(result.bitmap) ||
      mtmd_bitmap_get_data(result.bitmap)[0] != 255) {
    std::fprintf(stderr, "media helper did not preserve decoded image bytes\n");
    return 1;
  }
  mtmd_bitmap_free(result.bitmap);
  return 0;
}
