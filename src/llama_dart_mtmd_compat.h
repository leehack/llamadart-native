#pragma once

#include "mtmd-helper.h"
#include "llama_dart_wrapper.h"

#include <cstring>
#include <vector>

#if LLAMADART_MTMD_HELPER_HAS_EMBD_BATCH
// Keep the libllamadart callback ABI when v0.6.0 supplies a read-only view.
// Storage lives through the synchronous callback; M-RoPE remains section-major.
struct llama_dart_mtmd_callback_adapter {
  int32_t (*callback)(llama_batch, void *);
  void *user_data;

  static int32_t invoke(const mtmd_helper_embd_batch *view, void *opaque) {
    auto &adapter = *static_cast<llama_dart_mtmd_callback_adapter *>(opaque);
    std::vector<float> embeddings(
        view->embd,
        view->embd + static_cast<size_t>(view->n_tokens) * view->n_embd);
    std::vector<llama_pos> positions(
        view->pos,
        view->pos + static_cast<size_t>(view->n_tokens) * view->n_pos);
    std::vector<int32_t> counts(view->n_tokens, 1);
    llama_seq_id sequence = view->seq_id;
    std::vector<llama_seq_id *> sequences(view->n_tokens, &sequence);
    std::vector<int8_t> logits(view->n_tokens, 0);
    llama_batch batch{view->n_tokens,   nullptr,       embeddings.data(),
                      positions.data(), counts.data(), sequences.data(),
                      logits.data()};
    return adapter.callback(batch, adapter.user_data);
  }
};
#endif

static inline int32_t llama_dart_decode_image_chunk_compat(
    mtmd_context *ctx, llama_context *lctx, const mtmd_input_chunk *chunk,
    float *encoded_embd, llama_pos n_past, llama_seq_id seq_id, int32_t n_batch,
    llama_pos *new_n_past, int32_t (*callback)(llama_batch, void *),
    void *user_data) {
#if LLAMADART_MTMD_HELPER_HAS_EMBD_BATCH
  llama_dart_mtmd_callback_adapter adapter{callback, user_data};
  return mtmd_helper_decode_image_chunk(
      ctx, lctx, chunk, encoded_embd, n_past, seq_id, n_batch, new_n_past,
      callback == nullptr ? nullptr : llama_dart_mtmd_callback_adapter::invoke,
      &adapter);
#else
  return mtmd_helper_decode_image_chunk(ctx, lctx, chunk, encoded_embd, n_past,
                                        seq_id, n_batch, new_n_past, callback,
                                        user_data);
#endif
}

// v0.4.0 added helper options; retain builds against the published v0.3.0
// headers and use upstream's defaults when the options API is available.
static inline mtmd_helper_bitmap_wrapper
llama_dart_bitmap_from_buffer(mtmd_context *ctx, const unsigned char *buffer,
                              size_t size) {
#if LLAMADART_MTMD_HELPER_HAS_OPTIONS
  return mtmd_helper_bitmap_init_from_buf(ctx, buffer, size, false,
                                          mtmd_helper_init_opt_default());
#else
  return mtmd_helper_bitmap_init_from_buf(ctx, buffer, size, false);
#endif
}

static inline mtmd_helper_bitmap_wrapper
llama_dart_bitmap_from_file(mtmd_context *ctx, const char *fname) {
#if LLAMADART_MTMD_HELPER_HAS_OPTIONS
  return mtmd_helper_bitmap_init_from_file(ctx, fname, false,
                                           mtmd_helper_init_opt_default());
#else
  return mtmd_helper_bitmap_init_from_file(ctx, fname, false);
#endif
}

// Internal audio-frame policy, independent of the talker request temperature.
static inline mtmd_helper_gen_audio_inp llama_dart_audio_input_compat(
    llama_dart_tts_model_type model_type, const char *variant) {
  mtmd_helper_gen_audio_inp input{};
#if LLAMADART_MTMD_AUDIO_HAS_SPEED
  // Upstream rejects speed <= 0; the other delivery fields default to zero.
  input.speed = 1.0f;
#endif
#if LLAMADART_MTMD_AUDIO_HAS_TEMP
  // v1 adds an explicit frame/noise temperature; zero means greedy. LFM3
  // previously used mtmd_gen_inp_default, as did untuned Pocket variants.
  input.temp = -1.0f;
  if (model_type == LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS && variant) {
    // Preserve v0.9's reference-pack overrides, removed from v1's pack table.
    if (std::strcmp(variant, "english") == 0 ||
        std::strcmp(variant, "english_2026-04") == 0) {
      input.temp = 0.3f;
    } else if (std::strcmp(variant, "english_2026-01") == 0 ||
               std::strcmp(variant, "french_24l") == 0) {
      input.temp = 0.7f;
    }
  }
#else
  (void)model_type;
  (void)variant;
#endif
  return input;
}
