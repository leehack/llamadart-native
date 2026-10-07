#include "llama_dart_mtmd_compat.h"

#include <cassert>
#include <cstdio>

#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#if LLAMADART_MTMD_HELPER_HAS_EMBD_BATCH
struct callback_expectation {
  const mtmd_helper_embd_batch *view;
  int calls = 0;
};

static int32_t inspect_legacy_callback(llama_batch batch, void *opaque) {
  auto &expect = *static_cast<callback_expectation *>(opaque);
  const auto &view = *expect.view;
  ++expect.calls;
  assert(batch.n_tokens == view.n_tokens && batch.token == nullptr);
  for (int32_t i = 0; i < view.n_tokens; ++i) {
    assert(batch.n_seq_id[i] == 1 && batch.seq_id[i][0] == view.seq_id);
    assert(batch.logits[i] == 0);
    for (int32_t p = 0; p < view.n_pos; ++p) {
      assert(batch.pos[p * view.n_tokens + i] ==
             view.pos[p * view.n_tokens + i]);
    }
    for (int32_t e = 0; e < view.n_embd; ++e) {
      assert(batch.embd[i * view.n_embd + e] == view.embd[i * view.n_embd + e]);
    }
  }
  // Callback mutation must not write through upstream's const views.
  batch.embd[0] = -10;
  batch.pos[0] = -10;
  return 73;
}

static void test_callback_adapter() {
  const float embeddings[] = {1, 2, 3, 4, 5, 6};
  const llama_pos positions[] = {10, 11, 20, 21, 30, 31, 40, 41};
  for (int32_t n_pos : {1, 4}) {
    const mtmd_helper_embd_batch view{2, embeddings, 3, positions, n_pos, 7};
    callback_expectation expect{&view};
    llama_dart_mtmd_callback_adapter adapter{inspect_legacy_callback, &expect};
    assert(llama_dart_mtmd_callback_adapter::invoke(&view, &adapter) == 73);
    assert(expect.calls == 1);
    assert(embeddings[0] == 1 && positions[0] == 10);
  }
}
#endif

int main() {
#if LLAMADART_MTMD_HELPER_HAS_EMBD_BATCH
  test_callback_adapter();
#endif
  // A real one-pixel PPM exercises buffer forwarding and non-placeholder
  // decoding without a model. TTS rejects this non-audio result at its call
  // site.
  const unsigned char image[] = "P6\n1 1\n255\n\xff\x00\x00";
  auto result =
      llama_dart_bitmap_from_buffer(nullptr, image, sizeof(image) - 1);
  if (!result.bitmap || result.video_ctx ||
      mtmd_bitmap_is_audio(result.bitmap) ||
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
