#include "llama_dart_mtmd_compat.h"

#include <cassert>
#include <cstdio>
#include <type_traits>
#include <utility>

#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

template<class T, class = void> struct audio_has_temp : std::false_type {};
template<class T> struct audio_has_temp<T, std::void_t<decltype(std::declval<T>().temp)>>
    : std::true_type {};
static_assert(audio_has_temp<mtmd_helper_gen_audio_inp>::value ==
                  static_cast<bool>(LLAMADART_MTMD_AUDIO_HAS_TEMP),
              "audio temperature probe disagrees with the headers");

template<class T, class = void> struct audio_has_speed : std::false_type {};
template<class T> struct audio_has_speed<T, std::void_t<decltype(std::declval<T>().speed)>>
    : std::true_type {};
static_assert(audio_has_speed<mtmd_helper_gen_audio_inp>::value ==
                  static_cast<bool>(LLAMADART_MTMD_AUDIO_HAS_SPEED),
              "audio speed probe disagrees with the headers");

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
#if LLAMADART_MTMD_AUDIO_HAS_TEMP
  // Expected values come from v0.9's pack policy, not the new model defaults.
  for (const char *variant : {"english", "english_2026-04"}) {
    assert(llama_dart_audio_input_compat(LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS,
                                         variant).temp == 0.3f);
  }
  for (const char *variant : {"english_2026-01", "french_24l"}) {
    assert(llama_dart_audio_input_compat(LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS,
                                         variant).temp == 0.7f);
  }
  for (const char *variant : {static_cast<const char *>(nullptr), "", "unknown", "italian"}) {
    assert(llama_dart_audio_input_compat(LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS,
                                         variant).temp == -1.0f);
  }
  for (auto type : {LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO,
                    LLAMA_DART_TTS_MODEL_TYPE_QWEN3,
                    LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX}) {
    assert(llama_dart_audio_input_compat(type, "english").temp == -1.0f);
  }
#endif
#if LLAMADART_MTMD_AUDIO_HAS_SPEED
  // mtmd_helper_gen_audio_set_input rejects speed <= 0 with status 2, so a
  // zero-initialized input fails every TTS setup. Delivery controls other than
  // speed default to 0 (the model's own).
  for (auto type : {LLAMA_DART_TTS_MODEL_TYPE_NONE,
                    LLAMA_DART_TTS_MODEL_TYPE_QWEN3,
                    LLAMA_DART_TTS_MODEL_TYPE_LFM3_AUDIO,
                    LLAMA_DART_TTS_MODEL_TYPE_SOPRANO,
                    LLAMA_DART_TTS_MODEL_TYPE_CHATTERBOX,
                    LLAMA_DART_TTS_MODEL_TYPE_POCKET_TTS,
                    LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN}) {
    for (const char *variant : {static_cast<const char *>(nullptr), "english"}) {
      const auto defaults = mtmd_helper_gen_audio_inp_default();
      const auto input = llama_dart_audio_input_compat(type, variant);
      assert(input.speed == 1.0f);
      assert(input.speed == defaults.speed);
      assert(input.loudness == defaults.loudness);
      assert(input.cfg_scale == defaults.cfg_scale);
      assert(input.flow_steps == defaults.flow_steps);
    }
  }
#endif
  const auto input = llama_dart_audio_input_compat(LLAMA_DART_TTS_MODEL_TYPE_QWEN3, nullptr);
  assert(!input.stream && input.n_past == 0 && input.seq_id == 0);
  assert(!input.prompt && !input.speaker_ref && !input.lang);

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
