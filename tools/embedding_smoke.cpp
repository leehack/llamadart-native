// Opt-in real-weight qualification of guarded mixed processing against the
// upstream reference. --upstream-only keeps the direct mixed comparison mode.
#include "llama_dart_wrapper.h"
#include "mtmd.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message && *message ? message
                                                 : "native call failed");
  }
}

template <typename T> static auto tracked(T *pointer) {
  require(pointer != nullptr, llama_dart_last_error());
  return std::unique_ptr<T, void (*)(void *)>(pointer, llama_dart_exit_free);
}

static std::vector<float> normalized(std::vector<float> values, size_t length) {
  require(length <= values.size(), "invalid truncation length");
  values.resize(length);
  double sum = 0;
  for (const float value : values) {
    require(std::isfinite(value), "non-finite embedding");
    sum += double(value) * value;
  }
  require(sum > 0 && std::isfinite(sum), "zero or non-finite embedding norm");
  for (float &value : values) {
    value /= std::sqrt(sum);
  }
  return values;
}

static std::vector<float> output(llama_context *context) {
  const auto *model = llama_get_model(context);
  const int width = llama_model_n_embd_out(model);
  require(width == 768 && llama_model_n_embd_inp(model) == 512,
          "expected EmbeddingGemma-2 input 512 / output 768");
  const float *values = llama_get_embeddings_seq(context, 0);
  require(values != nullptr, "missing pooled sequence embedding");
  std::vector<float> result(values, values + width);
  for (const size_t length : {128, 256, 512, 768}) {
    const auto prefix = normalized(result, length);
    double sum = 0;
    for (const float value : prefix) {
      sum += double(value) * value;
    }
    require(std::abs(sum - 1) < 1e-5, "prefix normalization failed");
  }
  return normalized(result, width);
}

static float difference(const std::vector<float> &a,
                        const std::vector<float> &b) {
  require(a.size() == b.size(), "output dimensions differ");
  float result = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    result = std::max(result, std::abs(a[i] - b[i]));
  }
  return result;
}

static std::vector<float> text_embedding(llama_context *context, bool wrapper) {
  const auto *vocab = llama_model_get_vocab(llama_get_model(context));
  const std::string text = "task: search result | title: Paris | text: "
                           "The Eiffel Tower is in Paris.";
  std::vector<llama_token> tokens(128);
  const int count = llama_tokenize(vocab, text.data(), text.size(),
                                   tokens.data(), tokens.size(), true, true);
  require(count > 0 && count <= int(tokens.size()), "tokenization failed");
  auto batch = llama_batch_init(count, 0, 1);
  batch.n_tokens = count;
  for (int i = 0; i < count; ++i) {
    batch.token[i] = tokens[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = 1;
  }
  const int status = wrapper ? llama_dart_decode(context, batch)
                             : llama_decode(context, batch);
  llama_batch_free(batch);
  require(status == 0, "text decode failed");
  return output(context);
}

static std::vector<float> mixed_embedding(llama_context *context,
                                          mtmd_context *projector,
                                          const mtmd_bitmap *bitmap,
                                          bool wrapper) {
  const std::string prompt =
      std::string("task: search result | text: ") + mtmd_default_marker();
  mtmd_input_text text{prompt.data(), prompt.size(), true, true};
  mtmd::input_chunks_ptr chunks(mtmd_input_chunks_init());
  require((wrapper ? llama_dart_mtmd_tokenize(projector, chunks.get(), &text,
                                              &bitmap, 1)
                   : mtmd_tokenize(projector, chunks.get(), &text, &bitmap,
                                   1)) == 0,
          "media tokenization failed");
  const auto width = llama_model_n_embd_inp(llama_get_model(context));
  std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
      wrapper ? llama_dart_batch_ext_init(context)
              : llama_batch_ext_init(context),
      llama_batch_ext_free);
  require(batch != nullptr, "extended batch allocation failed");
  std::vector<std::vector<float>> media;
  llama_pos position = 0;
  bool saw_media = false;
  for (size_t c = 0; c < mtmd_input_chunks_size(chunks.get()); ++c) {
    const auto *chunk = mtmd_input_chunks_get(chunks.get(), c);
    size_t count = mtmd_input_chunk_get_n_tokens(chunk);
    const llama_token *tokens = nullptr;
    const float *embeddings = nullptr;
    if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
    } else {
      saw_media = true;
      require((wrapper ? llama_dart_mtmd_encode_chunk(projector, chunk)
                       : mtmd_encode_chunk(projector, chunk)) == 0,
              "media encode failed");
      const float *encoded = mtmd_get_output_embd(projector);
      require(encoded != nullptr, "missing media embeddings");
      // mtmd reuses its output storage on the next encode.
      media.emplace_back(encoded, encoded + count * width);
      embeddings = media.back().data();
    }
    require(count <= llama_n_ubatch(context) - size_t(position),
            "complete noncausal input must fit a single microbatch");
    for (size_t i = 0; i < count; ++i, ++position) {
      const int index =
          tokens
              ? llama_batch_ext_add_token(batch.get(), 0, tokens[i])
              : llama_batch_ext_add_embd(
                    batch.get(), 0, {embeddings + i * width, 1, size_t(width)});
      require(index >= 0, "mixed batch rejected an entry");
      require(llama_batch_ext_set_pos(batch.get(), index, &position),
              "mixed batch position rejected");
      require(llama_batch_ext_set_output_embd(batch.get(), index, true),
              "mixed batch output selection rejected");
    }
  }
  require(saw_media, "media input was silently discarded");
  require((wrapper ? llama_dart_process(context, LLAMA_PROCESS_TYPE_DECODE,
                                        batch.get())
                   : llama_process(context, LLAMA_PROCESS_TYPE_DECODE,
                                   batch.get())) == 0,
          "mixed decode failed");
  return output(context);
}

int main(int argc, char **argv) {
  const bool upstream_only =
      argc == 4 && std::string(argv[3]) == "--upstream-only";
  if (argc != 3 && !upstream_only) {
    std::fprintf(stderr, "Usage: %s model.gguf mmproj.gguf [--upstream-only]\n",
                 argv[0]);
    return 2;
  }
  llama_backend_init();
  try {
    std::printf("ABI context_params size=%zu; type_v=%zu; abort_callback=%zu; "
                "perf_context_data size=%zu\n",
                sizeof(llama_context_params),
                offsetof(llama_context_params, type_v),
                offsetof(llama_context_params, abort_callback),
                sizeof(llama_perf_context_data));
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
    std::printf("ABI context_params moe_cache_size=%zu\n",
                offsetof(llama_context_params, moe_cache_size));
#endif
    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    auto model =
        tracked(llama_dart_model_load_from_file(argv[1], model_params));
    auto params = llama_context_default_params();
    params.n_ctx = params.n_batch = params.n_ubatch = 2048;
    params.n_threads = params.n_threads_batch = 8;
    params.embeddings = true;
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    auto context = tracked(llama_dart_init_from_model(model.get(), params));
    auto reference = tracked(llama_dart_init_from_model(model.get(), params));
    const auto wrapped = text_embedding(context.get(), true);
    const auto direct = text_embedding(reference.get(), false);
    require(difference(wrapped, direct) < 1e-5,
            "wrapper text output differs from upstream");
    std::printf("PASS text: 768 dimensions; wrapper/upstream max delta %.8g\n",
                difference(wrapped, direct));
    auto media_params = mtmd_context_params_default();
    media_params.use_gpu = false;
    media_params.warmup = false;
    media_params.n_threads = 8;
    auto projector = tracked(
        llama_dart_mtmd_init_from_file(argv[2], model.get(), &media_params));
    require(mtmd_support_vision(projector.get()) &&
                mtmd_support_audio(projector.get()),
            "projector does not expose both image and audio input");
    const auto mixed = [&](const mtmd_bitmap *bitmap, const char *label) {
      const auto values = mixed_embedding(context.get(), projector.get(),
                                          bitmap, !upstream_only);
      if (!upstream_only) {
        const auto direct =
            mixed_embedding(reference.get(), projector.get(), bitmap, false);
        require(values == direct, "wrapper mixed output differs from upstream");
        std::printf(
            "PASS mixed %s: 768 dimensions; wrapper/upstream max delta %.8g\n",
            label, difference(values, direct));
      }
      return values;
    };
    std::vector<unsigned char> pixels(224 * 224 * 3, 0);
    mtmd::bitmap_ptr dark(mtmd_bitmap_init(224, 224, pixels.data()));
    for (size_t i = 0; i < pixels.size(); ++i) {
      pixels[i] = ((i / 3 / 224 / 16 + i / 3 % 224 / 16) % 2) ? 255 : 0;
    }
    mtmd::bitmap_ptr patterned(mtmd_bitmap_init(224, 224, pixels.data()));
    require(dark && patterned, "image allocation failed");
    const auto image_a = mixed(dark.get(), "dark image");
    const auto image_b = mixed(patterned.get(), "patterned image");
    require(difference(image_a, image_b) > 1e-4, "image content had no effect");
    const int rate = mtmd_get_audio_sample_rate(projector.get());
    require(rate == 16000, "expected 16 kHz audio");
    std::vector<float> samples(rate, 0);
    mtmd::bitmap_ptr silence(
        mtmd_bitmap_init_from_audio(samples.size(), samples.data()));
    for (size_t i = 0; i < samples.size(); ++i) {
      samples[i] = 0.2f * std::sin(2 * 3.141592653589793 * 440 * i / rate);
    }
    mtmd::bitmap_ptr tone(
        mtmd_bitmap_init_from_audio(samples.size(), samples.data()));
    require(silence && tone, "audio allocation failed");
    const auto audio_a = mixed(silence.get(), "silence");
    const auto audio_b = mixed(tone.get(), "tone");
    require(difference(audio_a, audio_b) > 1e-4, "audio content had no effect");
    require(difference(wrapped, text_embedding(context.get(), true)) < 1e-5,
            "media request contaminated the next text request");
    std::printf("PASS %s mixed image/audio: 768 dimensions; image delta "
                "%.8g; audio delta %.8g; text reuse stable\n",
                upstream_only ? "upstream-only" : "guarded",
                difference(image_a, image_b), difference(audio_a, audio_b));
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    llama_backend_free();
    return 1;
  }
  if (llama_dart_exit_tracked_count() != 0) {
    std::fprintf(stderr, "FAIL: tracked model/context leak\n");
    llama_backend_free();
    return 1;
  }
  llama_backend_free();
  return 0;
}
