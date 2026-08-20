#include "llama_dart_wrapper.h"
#include "llama_dart_speculative_compat.h"

#include "common.h"
#include "llama-ext.h"
#include "log.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "reasoning-budget.h"
#include "sampling.h"
#include "speculative.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif
#endif

// Global log level (0=none, 1=debug, 2=info, 3=warn, 4=error)
static std::atomic<int> g_dart_log_level{3}; // Default to WARN
// Track last non-CONT severity so continuation lines inherit proper level.
static std::atomic<int> g_last_non_cont_level{GGML_LOG_LEVEL_NONE};

static int llama_dart_common_log_verbosity(int level) {
  switch (level) {
  case 0:
    return -1;
  case 1:
    return LOG_LEVEL_DEBUG;
  case 2:
    return LOG_LEVEL_INFO;
  case 3:
    return LOG_LEVEL_WARN;
  case 4:
  default:
    return LOG_LEVEL_ERROR;
  }
}

struct llama_dart_mtp {
  llama_context *ctx_tgt = nullptr;
  llama_context *ctx_dft = nullptr;
  common_speculative *spec = nullptr;
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
};

struct llama_dart_ngram {
  common_speculative *spec = nullptr;
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
  bool has_last_draft = false;
};

struct llama_dart_speculative {
  llama_context *ctx_tgt = nullptr;
  llama_context *ctx_dft = nullptr;
  common_speculative *spec = nullptr;
  std::vector<uint32_t> target_output_layer_ids;
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
  std::vector<int8_t> process_output_mask;
  llama_dart_speculative_embedding_requirements embedding_requirements;
  bool caps_draft_process_outputs = false;
  bool has_last_draft = false;
};

struct llama_dart_tts {
  llama_context *llama = nullptr;
  mtmd_context *mtmd = nullptr;
  mtmd_helper_gen_audio *generator = nullptr;
  llama_sampler *sampler = nullptr;
  mtmd_bitmap *speaker = nullptr;
  std::atomic<bool> cancel_requested{false};
  llama_dart_tts_state state = LLAMA_DART_TTS_STATE_IDLE;
  llama_seq_id sequence_id = 0;
  bool owns_sequence = false;
  int32_t prompt_batch_size = 512;
  int32_t max_frames = 512;
  int32_t prompt_tokens_remaining = 0;
  int32_t frames_generated = 0;
  bool truncated = false;
  int32_t sample_rate = 0;
  int64_t sample_count = 0;
  std::vector<float> pcm;
  std::string language;
  std::string error;
};

static void llama_dart_tts_release_task_resources(llama_dart_tts *tts);

static llama_dart_tts_status llama_dart_tts_fail(
    llama_dart_tts *tts, llama_dart_tts_status status, const char *message) {
  if (tts != nullptr) {
    tts->state = LLAMA_DART_TTS_STATE_FAILED;
    tts->error = message != nullptr ? message : "unknown TTS error";
    llama_dart_tts_release_task_resources(tts);
  }
  return status;
}

static llama_dart_tts_status llama_dart_tts_error(
    llama_dart_tts *tts, llama_dart_tts_status status, const char *message) {
  if (tts != nullptr) {
    tts->error = message != nullptr ? message : "unknown TTS error";
  }
  return status;
}

static void llama_dart_tts_release_task_resources(llama_dart_tts *tts) {
  if (tts == nullptr) {
    return;
  }
  if (tts->speaker != nullptr) {
    mtmd_bitmap_free(tts->speaker);
    tts->speaker = nullptr;
  }
  if (tts->sampler != nullptr) {
    llama_sampler_free(tts->sampler);
    tts->sampler = nullptr;
  }
  if (tts->generator != nullptr) {
    mtmd_helper_gen_audio_reset(tts->generator);
  }
  if (tts->owns_sequence) {
    llama_memory_seq_rm(llama_get_memory(tts->llama), tts->sequence_id, 0, -1);
    tts->owns_sequence = false;
  }
}

static llama_dart_tts_model_type llama_dart_tts_model_type_from_upstream(
    mtmd_gen_audio_type type) {
  switch (type) {
  case MTMD_GEN_AUDIO_TYPE_NONE:
    return LLAMA_DART_TTS_MODEL_TYPE_NONE;
  case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
    return LLAMA_DART_TTS_MODEL_TYPE_QWEN3;
  default:
    return LLAMA_DART_TTS_MODEL_TYPE_UNKNOWN;
  }
}

static uint32_t llama_dart_tts_capabilities(const mtmd_context *mtmd,
                                            mtmd_gen_audio_type type) {
  switch (type) {
  case MTMD_GEN_AUDIO_TYPE_QWEN3TTS:
    return LLAMA_DART_TTS_CAPABILITY_LANGUAGE |
           (mtmd_support_audio(mtmd)
                ? LLAMA_DART_TTS_CAPABILITY_SPEAKER_REFERENCE
                : 0u);
  default:
    return 0;
  }
}

static llama_sampler *llama_dart_tts_sampler_init(
    const llama_dart_tts_request &request) {
  llama_sampler *sampler =
      llama_sampler_chain_init(llama_sampler_chain_default_params());
  if (sampler == nullptr) {
    return nullptr;
  }
  llama_sampler_chain_add(sampler, llama_sampler_init_top_k(request.top_k));
  llama_sampler_chain_add(sampler,
                          llama_sampler_init_top_p(request.top_p, 1));
  llama_sampler_chain_add(sampler,
                          llama_sampler_init_min_p(request.min_p, 1));
  llama_sampler_chain_add(
      sampler, llama_sampler_init_temp(request.temperature));
  llama_sampler_chain_add(sampler, llama_sampler_init_dist(request.seed));
  return sampler;
}

static int32_t llama_dart_tts_step_gen(
    int32_t (*step_gen)(mtmd_helper_gen_audio *, llama_token, const float *,
                        const float **),
    mtmd_helper_gen_audio *generator, llama_token sampled, const float *state,
    const float **next_state, bool *stop) {
  *stop = false;
  return step_gen(generator, sampled, state, next_state);
}

static int32_t llama_dart_tts_step_gen(
    int32_t (*step_gen)(mtmd_helper_gen_audio *, llama_token, const float *,
                        const float **, bool *),
    mtmd_helper_gen_audio *generator, llama_token sampled, const float *state,
    const float **next_state, bool *stop) {
  return step_gen(generator, sampled, state, next_state, stop);
}

static llama_dart_tts_status llama_dart_tts_finish_output(
    llama_dart_tts *tts) {
  int32_t sample_rate = 0;
  const char *data = nullptr;
  size_t data_len = 0;
  int64_t sample_count = 0;
  if (mtmd_helper_gen_audio_get_output(tts->generator, &sample_rate, &data,
                                       &data_len, &sample_count) != 0) {
    return llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                               "audio output conversion failed");
  }
  const bool sample_count_overflows =
      sample_count > 0 &&
      static_cast<uint64_t>(sample_count) >
          std::numeric_limits<size_t>::max() / sizeof(float);
  if (sample_rate <= 0 || sample_count <= 0 || sample_count_overflows ||
      data_len != static_cast<size_t>(sample_count) * sizeof(float) ||
      (data_len > 0 && data == nullptr)) {
    return llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                               "audio output metadata is invalid");
  }
  const float *samples = reinterpret_cast<const float *>(data);
  tts->pcm.clear();
  if (sample_count > 0) {
    tts->pcm.assign(samples, samples + sample_count);
  }
  tts->sample_rate = sample_rate;
  tts->sample_count = sample_count;
  tts->state = LLAMA_DART_TTS_STATE_COMPLETED;
  llama_dart_tts_release_task_resources(tts);
  return LLAMA_DART_TTS_STATUS_OK;
}

static void llama_dart_tts_write_progress(const llama_dart_tts *tts,
                                          llama_dart_tts_progress *out) {
  if (out == nullptr) {
    return;
  }
  out->struct_size = sizeof(*out);
  out->state = tts->state;
  out->prompt_tokens_remaining = tts->prompt_tokens_remaining;
  out->frames_generated = tts->frames_generated;
  out->truncated = tts->truncated;
}

#if defined(__APPLE__) && (TARGET_OS_IOS || TARGET_OS_TV || TARGET_OS_VISION)
__attribute__((constructor)) static void
llama_dart_configure_apple_mobile_environment() {
  // iOS-family devices can fail Metal runtime compilation when residency sets
  // increase memory pressure. Keep Metal enabled but disable that optimization.
  setenv("GGML_METAL_NO_RESIDENCY", "1", 0);
}
#endif

static void llama_dart_native_log_callback(ggml_log_level level,
                                           const char *text, void *user_data) {
  (void)user_data;
  const int configured_level = g_dart_log_level.load(std::memory_order_relaxed);
  // Explicitly suppress all native logs for `none`.
  if (configured_level <= 0) {
    return;
  }

  // ggml levels: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5.
  // CONT lines are continuations of the previous log message; they should
  // follow the previous message severity, not be treated as level 5.
  int effective_level;
  if (level == GGML_LOG_LEVEL_CONT) {
    effective_level = g_last_non_cont_level.load(std::memory_order_relaxed);
  } else {
    effective_level = static_cast<int>(level);
    g_last_non_cont_level.store(effective_level, std::memory_order_relaxed);
  }

  if (effective_level == GGML_LOG_LEVEL_NONE) {
    return;
  }

  if (effective_level >= configured_level) {
    fputs(text, stderr);
    fflush(stderr);
  }
}

static uint16_t llama_dart_uint16_or_default(int32_t value,
                                             uint16_t default_value) {
  if (value <= 0) {
    return default_value;
  }
  if (value > std::numeric_limits<uint16_t>::max()) {
    return std::numeric_limits<uint16_t>::max();
  }
  return static_cast<uint16_t>(value);
}

static bool llama_dart_type_mask_has(uint32_t type_mask,
                                     common_speculative_type type) {
  return (type_mask & (1u << static_cast<uint32_t>(type))) != 0;
}

static llama_batch llama_dart_cap_batch_outputs(llama_batch batch,
                                                std::vector<int8_t> &mask) {
  mask.clear();

  if (batch.n_tokens <= 0 || batch.logits == nullptr) {
    return batch;
  }

  int32_t output_count = 0;
  int32_t last_output = -1;
  for (int32_t i = 0; i < batch.n_tokens; ++i) {
    if (batch.logits[i] != 0) {
      ++output_count;
      last_output = i;
    }
  }

  if (output_count <= 1) {
    return batch;
  }

  mask.assign(static_cast<size_t>(batch.n_tokens), 0);
  mask[static_cast<size_t>(last_output)] = 1;
  batch.logits = mask.data();
  return batch;
}

static uint32_t llama_dart_type_mask_from_types(
    const std::vector<common_speculative_type> &types) {
  uint32_t result = 0;
  for (const auto type : types) {
    result |= (1u << static_cast<uint32_t>(type));
  }
  return result;
}

static std::vector<std::string>
llama_dart_split_speculative_type_names(const char *type_names) {
  std::vector<std::string> result;
  if (type_names == nullptr) {
    return result;
  }

  std::string value(type_names);
  size_t start = 0;
  while (start <= value.size()) {
    const size_t end = value.find(',', start);
    auto item = value.substr(
        start, end == std::string::npos ? std::string::npos : end - start);
    const auto first = item.find_first_not_of(" \t\r\n");
    if (first != std::string::npos) {
      const auto last = item.find_last_not_of(" \t\r\n");
      result.push_back(item.substr(first, last - first + 1));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return result;
}

static bool llama_dart_type_mask_has_draft_context(uint32_t type_mask) {
  return llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_MTP) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
}

static bool llama_dart_type_mask_has_non_mtp_draft_context(uint32_t type_mask) {
  return llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK);
}

static int llama_dart_count_draft_context_types(uint32_t type_mask) {
  int count = 0;
  if (llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE)) {
    ++count;
  }
  if (llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) {
    ++count;
  }
  if (llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) {
    ++count;
  }
  if (llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH)) {
    ++count;
  }
  if (llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK)) {
    ++count;
  }
  return count;
}

static std::vector<uint32_t> llama_dart_speculative_target_output_layer_ids(
    const llama_model *target_model, const llama_model *draft_model) {
  std::vector<uint32_t> result;
  if (target_model == nullptr || draft_model == nullptr) {
    return result;
  }

  const int32_t n_layer_tgt = llama_model_n_layer(target_model);
  const int32_t *target_layer_ids =
      llama_model_target_layer_ids(draft_model);
  const uint32_t target_layer_ids_n =
      llama_model_target_layer_ids_n(draft_model);
  result.reserve(target_layer_ids_n);
  for (uint32_t i = 0; i < target_layer_ids_n; ++i) {
    const int32_t layer_id = target_layer_ids[i];
    if (layer_id >= 0 && layer_id < n_layer_tgt) {
      result.push_back(static_cast<uint32_t>(layer_id));
    }
  }
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

static void llama_dart_reset_speculative_target_outputs(
    llama_context *target_context,
    const std::vector<uint32_t> &target_output_layer_ids) {
  if (target_context == nullptr) {
    return;
  }
  for (const uint32_t layer_id : target_output_layer_ids) {
    llama_set_embeddings_layer_inp(target_context, layer_id, false);
  }
  llama_set_embeddings(target_context, false);
  llama_set_embeddings_nextn(target_context, false, false);
}

static std::vector<common_speculative_type>
llama_dart_speculative_types_from_mask(uint32_t type_mask) {
  std::vector<common_speculative_type> result;
  if (type_mask == 0 ||
      llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_NONE)) {
    result.push_back(COMMON_SPECULATIVE_TYPE_NONE);
    return result;
  }

  for (int i = 1; i < static_cast<int>(COMMON_SPECULATIVE_TYPE_COUNT); ++i) {
    auto type = static_cast<common_speculative_type>(i);
    if (llama_dart_type_mask_has(type_mask, type)) {
      result.push_back(type);
    }
  }
  return result.empty()
      ? std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NONE}
      : result;
}

static std::vector<common_speculative_type>
llama_dart_speculative_types_from_params(
    const llama_dart_speculative_params &params) {
  const auto names = llama_dart_split_speculative_type_names(params.type_names);
  if (!names.empty()) {
    return common_speculative_types_from_names(names);
  }
  return llama_dart_speculative_types_from_mask(params.type_mask);
}

static void llama_dart_apply_ngram_map_params(
    common_params_speculative_ngram_map &dst,
    const llama_dart_speculative_params &src) {
  dst.size_n = llama_dart_uint16_or_default(src.ngram_size_n, dst.size_n);
  dst.size_m = llama_dart_uint16_or_default(src.ngram_size_m, dst.size_m);
  dst.min_hits = llama_dart_uint16_or_default(src.ngram_min_hits, dst.min_hits);
}

static common_params_speculative llama_dart_build_speculative_params(
    const llama_dart_speculative_params &src, llama_context *ctx_tgt,
    llama_context *ctx_dft) {
  common_params_speculative params;
  params.types = llama_dart_speculative_types_from_params(src);

  if (src.draft_token_max > 0) {
    params.draft.n_max = src.draft_token_max;
  }
  if (src.draft_token_min >= 0) {
    params.draft.n_min = src.draft_token_min;
  }
  if (src.draft_min_probability >= 0.0f) {
    params.draft.p_min = std::min(src.draft_min_probability, 1.0f);
  }
  if (src.draft_split_probability >= 0.0f) {
    params.draft.p_split = std::min(src.draft_split_probability, 1.0f);
  }
  params.draft.backend_sampling = src.backend_sampling;
  params.draft.ctx_tgt = ctx_tgt;
  params.draft.ctx_dft = ctx_dft;

  llama_dart_apply_ngram_map_params(params.ngram_simple, src);
  llama_dart_apply_ngram_map_params(params.ngram_map_k, src);
  llama_dart_apply_ngram_map_params(params.ngram_map_k4v, src);

  if (src.ngram_match > 0) {
    params.ngram_mod.n_match = src.ngram_match;
  }
  if (src.ngram_token_min >= 0) {
    params.ngram_mod.n_min = src.ngram_token_min;
  }
  if (src.ngram_token_max > 0) {
    params.ngram_mod.n_max = src.ngram_token_max;
  }
  if (src.ngram_cache_static_path != nullptr) {
    params.ngram_cache.lookup_cache_static = src.ngram_cache_static_path;
  }
  if (src.ngram_cache_dynamic_path != nullptr) {
    params.ngram_cache.lookup_cache_dynamic = src.ngram_cache_dynamic_path;
  }

  return params;
}

struct llama_dart_reasoning_budget {
  llama_sampler *budget = nullptr;
  llama_sampler *grammar = nullptr;
  bool pause_grammar_while_reasoning = false;
};

static bool llama_dart_reasoning_budget_should_apply_grammar(
    const llama_dart_reasoning_budget *sampler) {
  if (sampler == nullptr || sampler->grammar == nullptr) {
    return false;
  }
  if (!sampler->pause_grammar_while_reasoning) {
    return true;
  }

  const auto state = common_reasoning_budget_get_state(sampler->budget);
  return state == REASONING_BUDGET_IDLE || state == REASONING_BUDGET_DONE;
}

static bool llama_dart_reasoning_budget_matches(
    const llama_token *tokens, int32_t token_count, int32_t start,
    const std::vector<llama_token> &sequence) {
  if (tokens == nullptr || sequence.empty() || start < 0 ||
      start > token_count ||
      sequence.size() > static_cast<size_t>(token_count - start)) {
    return false;
  }

  for (size_t index = 0; index < sequence.size(); ++index) {
    if (tokens[start + static_cast<int32_t>(index)] != sequence[index]) {
      return false;
    }
  }
  return true;
}

static common_reasoning_budget_state
llama_dart_reasoning_budget_initial_state(
    const std::vector<llama_token> &start_tokens,
    const std::vector<llama_token> &end_tokens,
    const llama_token *prompt_tokens, int32_t prompt_token_count) {
  auto state = REASONING_BUDGET_IDLE;
  for (int32_t index = 0; index < prompt_token_count;) {
    if (llama_dart_reasoning_budget_matches(
            prompt_tokens, prompt_token_count, index, start_tokens)) {
      state = REASONING_BUDGET_COUNTING;
      index += static_cast<int32_t>(start_tokens.size());
      continue;
    }
    if (llama_dart_reasoning_budget_matches(
            prompt_tokens, prompt_token_count, index, end_tokens)) {
      state = REASONING_BUDGET_IDLE;
      index += static_cast<int32_t>(end_tokens.size());
      continue;
    }
    ++index;
  }
  return state;
}

template <typename ReasoningBudgetInit>
static llama_sampler *llama_dart_reasoning_budget_init_compat(
    ReasoningBudgetInit init, const llama_vocab *vocab,
    const llama_tokens &start_tokens, const llama_tokens &end_tokens,
    const llama_tokens &forced_tokens, int32_t budget_tokens,
    common_reasoning_budget_state initial_state) {
  using llama_token_sequences = std::vector<llama_tokens>;
  if constexpr (std::is_invocable_r_v<
                    llama_sampler *, ReasoningBudgetInit, const llama_vocab *,
                    const llama_token_sequences &,
                    const llama_token_sequences &, const llama_tokens &,
                    int32_t, common_reasoning_budget_state>) {
    return init(vocab, llama_token_sequences{start_tokens},
                llama_token_sequences{end_tokens}, forced_tokens, budget_tokens,
                initial_state);
  } else {
    return init(vocab, start_tokens, end_tokens, forced_tokens, budget_tokens,
                initial_state);
  }
}

static const char *
llama_dart_reasoning_budget_name(const struct llama_sampler * /*sampler*/) {
  return "llamadart-reasoning-budget";
}

static void llama_dart_reasoning_budget_accept(struct llama_sampler *sampler,
                                               llama_token token) {
  auto *context = static_cast<llama_dart_reasoning_budget *>(sampler->ctx);
  const bool accept_grammar =
      llama_dart_reasoning_budget_should_apply_grammar(context);

  llama_sampler_accept(context->budget, token);
  if (accept_grammar) {
    llama_sampler_accept(context->grammar, token);
  }
}

static void llama_dart_reasoning_budget_apply(
    struct llama_sampler *sampler, llama_token_data_array *candidates) {
  auto *context = static_cast<llama_dart_reasoning_budget *>(sampler->ctx);
  llama_sampler_apply(context->budget, candidates);
  if (llama_dart_reasoning_budget_should_apply_grammar(context)) {
    llama_sampler_apply(context->grammar, candidates);
  }
}

static void llama_dart_reasoning_budget_reset(struct llama_sampler *sampler) {
  auto *context = static_cast<llama_dart_reasoning_budget *>(sampler->ctx);
  llama_sampler_reset(context->budget);
  if (context->grammar != nullptr) {
    llama_sampler_reset(context->grammar);
  }
}

static struct llama_sampler *
llama_dart_reasoning_budget_clone(const struct llama_sampler *sampler);

static void llama_dart_reasoning_budget_free(struct llama_sampler *sampler) {
  auto *context = static_cast<llama_dart_reasoning_budget *>(sampler->ctx);
  if (context == nullptr) {
    return;
  }

  if (context->budget != nullptr) {
    llama_sampler_free(context->budget);
  }
  if (context->grammar != nullptr) {
    llama_sampler_free(context->grammar);
  }
  delete context;
}

static struct llama_sampler_i llama_dart_reasoning_budget_interface = {
    /* .name              = */ llama_dart_reasoning_budget_name,
    /* .accept            = */ llama_dart_reasoning_budget_accept,
    /* .apply             = */ llama_dart_reasoning_budget_apply,
    /* .reset             = */ llama_dart_reasoning_budget_reset,
    /* .clone             = */ llama_dart_reasoning_budget_clone,
    /* .free              = */ llama_dart_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
};

static struct llama_sampler *
llama_dart_reasoning_budget_clone(const struct llama_sampler *sampler) {
  const auto *context =
      static_cast<const llama_dart_reasoning_budget *>(sampler->ctx);
  if (context == nullptr || context->budget == nullptr) {
    return nullptr;
  }

  auto *budget_clone = llama_sampler_clone(context->budget);
  auto *grammar_clone = context->grammar == nullptr
      ? nullptr
      : llama_sampler_clone(context->grammar);
  if (budget_clone == nullptr ||
      (context->grammar != nullptr && grammar_clone == nullptr)) {
    if (budget_clone != nullptr) {
      llama_sampler_free(budget_clone);
    }
    if (grammar_clone != nullptr) {
      llama_sampler_free(grammar_clone);
    }
    return nullptr;
  }

  auto *clone = new llama_dart_reasoning_budget{
      /* .budget                        = */ budget_clone,
      /* .grammar                       = */ grammar_clone,
      /* .pause_grammar_while_reasoning = */
          context->pause_grammar_while_reasoning,
  };
  auto *result =
      llama_sampler_init(&llama_dart_reasoning_budget_interface, clone);
  if (result == nullptr) {
    llama_sampler_free(clone->budget);
    if (clone->grammar != nullptr) {
      llama_sampler_free(clone->grammar);
    }
    delete clone;
  }
  return result;
}

extern "C" {

LLAMADART_API void llama_dart_set_log_level(int level) {
  if (level < 0) {
    level = 0;
  } else if (level > 4) {
    level = 4;
  }

  g_dart_log_level.store(level, std::memory_order_relaxed);
  g_last_non_cont_level.store(GGML_LOG_LEVEL_NONE, std::memory_order_relaxed);
  common_log_set_verbosity_thold(llama_dart_common_log_verbosity(level));
  // Set callbacks every time to ensure they are active
  llama_log_set(llama_dart_native_log_callback, nullptr);
  ggml_log_set(llama_dart_native_log_callback, nullptr);
}

LLAMADART_API uint32_t llama_dart_tts_api_version(void) {
  return LLAMA_DART_TTS_API_VERSION;
}

LLAMADART_API struct llama_dart_tts_request
llama_dart_tts_request_default(void) {
  llama_dart_tts_request request{};
  request.struct_size = sizeof(request);
  request.sequence_id = 0;
  request.prompt_batch_size = 512;
  request.max_frames = 512;
  request.top_k = 40;
  request.top_p = 0.95f;
  request.min_p = 0.0f;
  request.temperature = 0.8f;
  request.seed = LLAMA_DEFAULT_SEED;
  return request;
}

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_info(
    const struct mtmd_context *mtmd, struct llama_dart_tts_info *out_info) {
  if (mtmd == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const mtmd_gen_audio_info upstream = mtmd_gen_audio_get_info(mtmd);
  out_info->api_version = LLAMA_DART_TTS_API_VERSION;
  out_info->model_type =
      llama_dart_tts_model_type_from_upstream(upstream.type);
  out_info->capabilities = llama_dart_tts_capabilities(mtmd, upstream.type);
  const bool supported = upstream.type == MTMD_GEN_AUDIO_TYPE_QWEN3TTS;
  out_info->sample_rate = supported ? upstream.sample_rate : 0;
  out_info->channels = supported ? 1 : 0;
  return !supported
             ? LLAMA_DART_TTS_STATUS_UNSUPPORTED
             : LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API struct llama_dart_tts *llama_dart_tts_init(
    struct llama_context *llama, struct mtmd_context *mtmd,
    enum llama_dart_tts_status *out_status) {
  if (out_status != nullptr) {
    *out_status = LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const mtmd_gen_audio_type type =
      mtmd == nullptr ? MTMD_GEN_AUDIO_TYPE_NONE
                      : mtmd_gen_audio_get_info(mtmd).type;
  if (llama == nullptr || mtmd == nullptr ||
      type != MTMD_GEN_AUDIO_TYPE_QWEN3TTS) {
    if (out_status != nullptr && llama != nullptr && mtmd != nullptr) {
      *out_status = LLAMA_DART_TTS_STATUS_UNSUPPORTED;
    }
    return nullptr;
  }
  mtmd_helper_gen_audio *generator = mtmd_helper_gen_audio_init(llama, mtmd);
  if (generator == nullptr) {
    if (out_status != nullptr) {
      *out_status = LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR;
    }
    return nullptr;
  }
  auto *tts = new llama_dart_tts();
  tts->llama = llama;
  tts->mtmd = mtmd;
  tts->generator = generator;
  if (out_status != nullptr) {
    *out_status = LLAMA_DART_TTS_STATUS_OK;
  }
  return tts;
}

LLAMADART_API void llama_dart_tts_free(struct llama_dart_tts *tts) {
  if (tts == nullptr) {
    return;
  }
  llama_dart_tts_release_task_resources(tts);
  mtmd_helper_gen_audio_free(tts->generator);
  tts->generator = nullptr;
  delete tts;
}

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start(
    struct llama_dart_tts *tts,
    const struct llama_dart_tts_request *request) {
  if (tts == nullptr) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state == LLAMA_DART_TTS_STATE_PROCESSING_PROMPT ||
      tts->state == LLAMA_DART_TTS_STATE_GENERATING) {
    return llama_dart_tts_error(tts, LLAMA_DART_TTS_STATUS_INVALID_STATE,
                                "a TTS task is already active");
  }
  if (request == nullptr || request->struct_size < sizeof(*request) ||
      request->text == nullptr ||
      request->text_length == 0 || request->prompt_batch_size <= 0 ||
      request->max_frames <= 0 || request->sequence_id < 0 ||
      request->top_k < 0 || !std::isfinite(request->top_p) ||
      request->top_p < 0.0f || request->top_p > 1.0f ||
      !std::isfinite(request->min_p) || request->min_p < 0.0f ||
      request->min_p > 1.0f || !std::isfinite(request->temperature) ||
      request->temperature < 0.0f ||
      (request->speaker_audio_length > 0 &&
       request->speaker_audio == nullptr)) {
    return llama_dart_tts_error(tts, LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT,
                                "invalid TTS request");
  }

  llama_dart_tts_release_task_resources(tts);
  tts->pcm.clear();
  tts->sample_rate = 0;
  tts->sample_count = 0;
  tts->frames_generated = 0;
  tts->prompt_tokens_remaining = 0;
  tts->truncated = false;
  tts->error.clear();
  tts->cancel_requested.store(false, std::memory_order_release);
  tts->sequence_id = request->sequence_id;
  tts->prompt_batch_size = request->prompt_batch_size;
  tts->max_frames = request->max_frames;
  tts->language = request->language != nullptr ? request->language : "";

  llama_memory_seq_rm(llama_get_memory(tts->llama), tts->sequence_id, 0, -1);
  tts->owns_sequence = true;
  if (request->speaker_audio_length > 0) {
    mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
        tts->mtmd, request->speaker_audio, request->speaker_audio_length, false);
    if (wrapper.bitmap == nullptr || !mtmd_bitmap_is_audio(wrapper.bitmap)) {
      if (wrapper.bitmap != nullptr) {
        mtmd_bitmap_free(wrapper.bitmap);
      }
      return llama_dart_tts_fail(
          tts, LLAMA_DART_TTS_STATUS_SPEAKER_DECODE_FAILED,
          "speaker reference audio could not be decoded");
    }
    tts->speaker = wrapper.bitmap;
  }

  tts->sampler = llama_dart_tts_sampler_init(*request);
  if (tts->sampler == nullptr) {
    return llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                               "sampler initialization failed");
  }

  mtmd_helper_gen_audio_inp input{};
  input.seq_id = tts->sequence_id;
  input.prompt = request->text;
  input.prompt_len = request->text_length;
  input.speaker_ref = tts->speaker;
  input.lang = tts->language.empty() ? nullptr : tts->language.c_str();
  input.top_k = request->top_k;
  input.top_p = request->top_p;
  input.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
  if (mtmd_helper_gen_audio_set_input(tts->generator, &input) != 0) {
    return llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                               "TTS input setup failed");
  }
  tts->state = LLAMA_DART_TTS_STATE_PROCESSING_PROMPT;
  return LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_step(
    struct llama_dart_tts *tts,
    struct llama_dart_tts_progress *out_progress) {
  if (tts == nullptr || out_progress == nullptr ||
      out_progress->struct_size < sizeof(*out_progress)) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const bool active = tts->state == LLAMA_DART_TTS_STATE_PROCESSING_PROMPT ||
                      tts->state == LLAMA_DART_TTS_STATE_GENERATING;
  if (active && tts->cancel_requested.load(std::memory_order_acquire)) {
    tts->state = LLAMA_DART_TTS_STATE_CANCELLED;
    tts->error = "TTS task cancelled";
    llama_dart_tts_release_task_resources(tts);
    llama_dart_tts_write_progress(tts, out_progress);
    return LLAMA_DART_TTS_STATUS_CANCELLED;
  }
  if (tts->state == LLAMA_DART_TTS_STATE_PROCESSING_PROMPT) {
    const int32_t remaining = mtmd_helper_gen_audio_step_prompt(
        tts->generator, tts->prompt_batch_size);
    if (remaining < 0) {
      const auto status = llama_dart_tts_fail(
          tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
          "TTS prompt processing failed");
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    tts->prompt_tokens_remaining = remaining;
    if (remaining == 0) {
      tts->state = LLAMA_DART_TTS_STATE_GENERATING;
    }
    llama_dart_tts_write_progress(tts, out_progress);
    return LLAMA_DART_TTS_STATUS_OK;
  }
  if (tts->state == LLAMA_DART_TTS_STATE_GENERATING) {
    if (tts->frames_generated >= tts->max_frames) {
      tts->truncated = true;
      const auto status = llama_dart_tts_finish_output(tts);
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    const llama_token sampled =
        llama_sampler_sample(tts->sampler, tts->llama, -1);
    if (sampled == LLAMA_TOKEN_NULL) {
      const auto status = llama_dart_tts_fail(
          tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
          "TTS token sampling failed");
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    const llama_vocab *vocab = llama_model_get_vocab(
        llama_get_model(tts->llama));
    if (vocab == nullptr) {
      const auto status = llama_dart_tts_fail(
          tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
          "TTS vocabulary is unavailable");
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    if (llama_vocab_is_eog(vocab, sampled)) {
      const auto status = llama_dart_tts_finish_output(tts);
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    const float *state = llama_get_embeddings_ith(tts->llama, -1);
    const float *next_state = nullptr;
    bool stop = false;
    if (state == nullptr ||
        llama_dart_tts_step_gen(&mtmd_helper_gen_audio_step_gen,
                                tts->generator, sampled, state, &next_state,
                                &stop) != 0) {
      const auto status = llama_dart_tts_fail(
          tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
          "TTS generation step failed");
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    if (next_state != nullptr) {
      ++tts->frames_generated;
    }
    if (stop || next_state == nullptr) {
      const auto status = llama_dart_tts_finish_output(tts);
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    llama_dart_tts_write_progress(tts, out_progress);
    return LLAMA_DART_TTS_STATUS_OK;
  }
  llama_dart_tts_write_progress(tts, out_progress);
  return tts->state == LLAMA_DART_TTS_STATE_CANCELLED
             ? LLAMA_DART_TTS_STATUS_CANCELLED
             : LLAMA_DART_TTS_STATUS_INVALID_STATE;
}

LLAMADART_API void llama_dart_tts_cancel(struct llama_dart_tts *tts) {
  if (tts != nullptr) {
    tts->cancel_requested.store(true, std::memory_order_release);
  }
}

LLAMADART_API enum llama_dart_tts_status
llama_dart_tts_reset(struct llama_dart_tts *tts) {
  if (tts == nullptr) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  llama_dart_tts_release_task_resources(tts);
  tts->cancel_requested.store(false, std::memory_order_release);
  tts->state = LLAMA_DART_TTS_STATE_IDLE;
  tts->prompt_tokens_remaining = 0;
  tts->frames_generated = 0;
  tts->truncated = false;
  tts->sample_rate = 0;
  tts->sample_count = 0;
  tts->pcm.clear();
  tts->language.clear();
  tts->error.clear();
  return LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_output_info(
    const struct llama_dart_tts *tts,
    struct llama_dart_tts_output_info *out_info) {
  if (tts == nullptr || out_info == nullptr ||
      out_info->struct_size < sizeof(*out_info)) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state != LLAMA_DART_TTS_STATE_COMPLETED) {
    return LLAMA_DART_TTS_STATUS_INVALID_STATE;
  }
  out_info->sample_rate = tts->sample_rate;
  out_info->channels = 1;
  out_info->sample_count = tts->sample_count;
  return LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_read_pcm(
    const struct llama_dart_tts *tts, int64_t sample_offset,
    float *out_samples, size_t out_capacity, size_t *out_count) {
  if (tts == nullptr || out_count == nullptr || sample_offset < 0) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state != LLAMA_DART_TTS_STATE_COMPLETED) {
    return LLAMA_DART_TTS_STATUS_INVALID_STATE;
  }
  if (static_cast<uint64_t>(sample_offset) >
      static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const size_t offset = static_cast<size_t>(sample_offset);
  if (offset > tts->pcm.size()) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const size_t remaining = tts->pcm.size() - offset;
  if (out_samples == nullptr) {
    *out_count = remaining;
    return LLAMA_DART_TTS_STATUS_OK;
  }
  const size_t count = std::min(remaining, out_capacity);
  if (count > 0) {
    std::copy_n(tts->pcm.data() + offset, count, out_samples);
  }
  *out_count = count;
  return LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API const char *
llama_dart_tts_last_error(const struct llama_dart_tts *tts) {
  return tts == nullptr ? "invalid TTS handle" : tts->error.c_str();
}

LLAMADART_API struct llama_sampler *llama_dart_sampler_init_reasoning_budget(
    const struct llama_vocab *vocab, const char *start_tag, const char *end_tag,
    const char *forced_message, int32_t budget_tokens,
    bool pause_grammar_while_reasoning,
    struct llama_sampler *grammar_sampler, const llama_token *prompt_tokens,
    int32_t prompt_token_count) {
  if (vocab == nullptr || start_tag == nullptr || end_tag == nullptr ||
      start_tag[0] == '\0' || end_tag[0] == '\0' || budget_tokens < 0 ||
      prompt_token_count < 0 ||
      (prompt_token_count > 0 && prompt_tokens == nullptr)) {
    return nullptr;
  }

  const auto start_tokens = common_tokenize(vocab, start_tag, false, true);
  const auto end_tokens = common_tokenize(vocab, end_tag, false, true);
  std::string forced_tokens_text =
      forced_message == nullptr ? "" : forced_message;
  forced_tokens_text += end_tag;
  const auto forced_tokens =
      common_tokenize(vocab, forced_tokens_text, false, true);
  if (start_tokens.empty() || end_tokens.empty() || forced_tokens.empty()) {
    return nullptr;
  }

  const auto initial_state = llama_dart_reasoning_budget_initial_state(
      start_tokens, end_tokens, prompt_tokens, prompt_token_count);
  auto *budget_sampler = llama_dart_reasoning_budget_init_compat(
      &common_reasoning_budget_init, vocab, start_tokens, end_tokens,
      forced_tokens, budget_tokens, initial_state);
  if (budget_sampler == nullptr) {
    return nullptr;
  }

  auto *context = new llama_dart_reasoning_budget{
      /* .budget                        = */ budget_sampler,
      /* .grammar                       = */ grammar_sampler,
      /* .pause_grammar_while_reasoning = */ pause_grammar_while_reasoning,
  };
  auto *result =
      llama_sampler_init(&llama_dart_reasoning_budget_interface, context);
  if (result == nullptr) {
    llama_sampler_free(context->budget);
    // Keep grammar_sampler owned by the caller on initialization failure. The
    // Dart bridge frees it when this function returns nullptr; ownership moves
    // to the composite sampler only after a non-null result is returned.
    delete context;
  }
  return result;
}

LLAMADART_API struct llama_dart_speculative *llama_dart_speculative_init(
    struct llama_model *target_model, struct llama_model *draft_model,
    struct llama_context *target_context,
    struct llama_context_params context_params,
    const struct llama_dart_speculative_params *dart_params) {
  if (target_context == nullptr) {
    return nullptr;
  }

  llama_dart_speculative_params default_params{};
  const llama_dart_speculative_params &params_input =
      dart_params == nullptr ? default_params : *dart_params;

  std::vector<common_speculative_type> types;
  try {
    types = llama_dart_speculative_types_from_params(params_input);
  } catch (const std::exception &e) {
    LOG_WRN("%s: failed to resolve speculative types: %s\n", __func__,
            e.what());
    return nullptr;
  } catch (...) {
    LOG_WRN("%s: failed to resolve speculative types\n", __func__);
    return nullptr;
  }
  const uint32_t type_mask = llama_dart_type_mask_from_types(types);
  if (type_mask == 0 ||
      llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_NONE)) {
    return nullptr;
  }

  if (llama_dart_count_draft_context_types(type_mask) > 1) {
    LOG_WRN("%s: selected speculative types require more than one draft "
            "context; choose at most one draft model strategy\n",
            __func__);
    return nullptr;
  }

  common_params_speculative params =
      llama_dart_build_speculative_params(params_input, target_context,
                                          nullptr);

  llama_context *ctx_dft = nullptr;
  if (llama_dart_type_mask_has_draft_context(type_mask)) {
    const bool has_mtp =
        llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
    const bool needs_external_draft =
        llama_dart_type_mask_has_non_mtp_draft_context(type_mask);

    llama_model *resolved_draft_model = draft_model;
    if (resolved_draft_model == nullptr && has_mtp && !needs_external_draft) {
      resolved_draft_model = target_model;
    }
    if (resolved_draft_model == nullptr) {
      LOG_WRN("%s: draft model is required for selected speculative types\n",
              __func__);
      return nullptr;
    }

    if (has_mtp) {
      context_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    context_params.n_seq_max = 1;
    context_params.n_rs_seq = 0;
    llama_dart_apply_speculative_draft_output_limits(
        context_params, types, params.draft.n_max,
        params.draft.backend_sampling);
    context_params.embeddings = false;
    context_params.ctx_other = target_context;

    ctx_dft = llama_init_from_model(resolved_draft_model, context_params);
    if (ctx_dft == nullptr) {
      LOG_WRN("%s: failed to create speculative draft context\n", __func__);
      return nullptr;
    }
  }

  params.draft.ctx_dft = ctx_dft;
  const auto target_output_layer_ids =
      llama_dart_speculative_target_output_layer_ids(
          target_model, ctx_dft == nullptr ? nullptr : llama_get_model(ctx_dft));

  common_speculative *spec = nullptr;
  try {
    spec = common_speculative_init(params, 1);
  } catch (const std::exception &e) {
    LOG_WRN("%s: failed to initialize common_speculative: %s\n", __func__,
            e.what());
  } catch (...) {
    LOG_WRN("%s: failed to initialize common_speculative\n", __func__);
  }
  if (spec == nullptr) {
    if (ctx_dft != nullptr) {
      llama_free(ctx_dft);
    }
    llama_dart_reset_speculative_target_outputs(target_context,
                                                 target_output_layer_ids);
    return nullptr;
  }

  auto *speculative = new llama_dart_speculative();
  speculative->ctx_tgt = target_context;
  speculative->ctx_dft = ctx_dft;
  speculative->spec = spec;
  speculative->target_output_layer_ids = target_output_layer_ids;
  speculative->embedding_requirements =
      llama_dart_speculative_embedding_requirements_for(types);
  speculative->caps_draft_process_outputs =
      llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
  return speculative;
}

LLAMADART_API void
llama_dart_speculative_free(struct llama_dart_speculative *speculative) {
  if (speculative == nullptr) {
    return;
  }
  if (speculative->spec != nullptr) {
    common_speculative_free(speculative->spec);
    speculative->spec = nullptr;
  }
  if (speculative->ctx_tgt != nullptr) {
    llama_dart_reset_speculative_target_outputs(
        speculative->ctx_tgt, speculative->target_output_layer_ids);
  }
  if (speculative->ctx_dft != nullptr) {
    llama_free(speculative->ctx_dft);
    speculative->ctx_dft = nullptr;
  }
  delete speculative;
}

LLAMADART_API struct llama_context *
llama_dart_speculative_get_draft_context(
    struct llama_dart_speculative *speculative) {
  if (speculative == nullptr) {
    return nullptr;
  }
  return speculative->ctx_dft;
}

LLAMADART_API bool
llama_dart_speculative_need_embd(struct llama_dart_speculative *speculative) {
  return speculative != nullptr &&
         speculative->embedding_requirements.need_embd;
}

LLAMADART_API bool llama_dart_speculative_need_embd_nextn(
    struct llama_dart_speculative *speculative) {
  return speculative != nullptr &&
         speculative->embedding_requirements.need_embd_nextn;
}

LLAMADART_API bool llama_dart_speculative_begin(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    const llama_token *prompt, int32_t prompt_count) {
  if (speculative == nullptr || speculative->spec == nullptr ||
      prompt_count < 0 || seq_id != 0) {
    return false;
  }

  speculative->prompt.clear();
  speculative->has_last_draft = false;
  if (prompt != nullptr && prompt_count > 0) {
    speculative->prompt.assign(prompt, prompt + prompt_count);
  }

  common_speculative_begin(speculative->spec, seq_id, speculative->prompt);
  return true;
}

LLAMADART_API bool llama_dart_speculative_process_batch(
    struct llama_dart_speculative *speculative, struct llama_batch batch) {
  if (speculative == nullptr || speculative->spec == nullptr) {
    return false;
  }
  if (speculative->caps_draft_process_outputs) {
    batch = llama_dart_cap_batch_outputs(batch, speculative->process_output_mask);
  }
  return common_speculative_process(speculative->spec, batch);
}

LLAMADART_API int32_t llama_dart_speculative_draft(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last, const llama_token *prompt,
    int32_t prompt_count, int32_t draft_token_max, llama_token *out_tokens,
    int32_t out_capacity) {
  if (speculative == nullptr || speculative->spec == nullptr ||
      out_tokens == nullptr || out_capacity < 0 || prompt_count < 0 ||
      draft_token_max <= 0 || seq_id != 0) {
    if (speculative != nullptr) {
      speculative->has_last_draft = false;
    }
    return -1;
  }

  speculative->prompt.clear();
  if (prompt != nullptr && prompt_count > 0) {
    speculative->prompt.assign(prompt, prompt + prompt_count);
  }
  speculative->draft.clear();
  speculative->has_last_draft = false;
  speculative->draft.reserve(static_cast<size_t>(draft_token_max));

  common_speculative_get_draft_params(speculative->spec, seq_id) = {
      /* .drafting = */ true,
      /* .n_max    = */ draft_token_max,
      /* .n_past   = */ n_past,
      /* .id_last  = */ id_last,
      /* .prompt   = */ &speculative->prompt,
      /* .result   = */ &speculative->draft,
  };

  common_speculative_draft(speculative->spec);

  const int32_t count = std::min<int32_t>(
      static_cast<int32_t>(speculative->draft.size()),
      std::min(draft_token_max, out_capacity));
  speculative->has_last_draft = count > 0;
  for (int32_t i = 0; i < count; ++i) {
    out_tokens[i] = speculative->draft[static_cast<size_t>(i)];
  }
  return count;
}

LLAMADART_API void llama_dart_speculative_accept(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    uint16_t accepted_count) {
  if (speculative == nullptr || speculative->spec == nullptr || seq_id != 0) {
    return;
  }
  if (!speculative->has_last_draft) {
    return;
  }
  common_speculative_accept(speculative->spec, seq_id, accepted_count);
  speculative->has_last_draft = false;
}

static struct llama_dart_mtp *llama_dart_mtp_init_impl(
    struct llama_model *draft_model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  if (draft_model == nullptr || ctx_tgt == nullptr) {
    if (draft_model != nullptr || ctx_tgt != nullptr) {
      LOG_WRN("%s: missing draft model or target context\n", __func__);
    }
    return nullptr;
  }

  if (draft_token_max <= 0) {
    draft_token_max = 1;
  }
  if (draft_token_min < 0) {
    draft_token_min = 0;
  }
  if (min_probability < 0.0f) {
    min_probability = 0.0f;
  } else if (min_probability > 1.0f) {
    min_probability = 1.0f;
  }

  ctx_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
  ctx_params.n_seq_max = 1;
  ctx_params.n_rs_seq = 0;
  ctx_params.n_outputs_max = 1;
  ctx_params.embeddings = false;
  ctx_params.ctx_other = ctx_tgt;

  llama_context *ctx_dft = llama_init_from_model(draft_model, ctx_params);
  if (ctx_dft == nullptr) {
    LOG_WRN("%s: failed to create MTP draft context\n", __func__);
    return nullptr;
  }

  const auto tgt_seq_rm_type = common_context_can_seq_rm(ctx_tgt);
  const auto dft_seq_rm_type = common_context_can_seq_rm(ctx_dft);
  const bool tgt_can_rollback =
      tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
      tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
  const bool dft_can_rollback =
      dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
      dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;
  if (!tgt_can_rollback || !dft_can_rollback) {
    LOG_WRN("%s: unsupported seq_rm type for first MTP implementation: target=%d, draft=%d\n",
            __func__, (int) tgt_seq_rm_type, (int) dft_seq_rm_type);
    llama_free(ctx_dft);
    return nullptr;
  }

  common_params_speculative params;
  params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_MTP};
  params.draft.ctx_tgt = ctx_tgt;
  params.draft.ctx_dft = ctx_dft;
  params.draft.n_max = draft_token_max;
  params.draft.n_min = draft_token_min;
  params.draft.p_min = min_probability;
  params.draft.backend_sampling = backend_sampling;

  common_speculative *spec = common_speculative_init(params, 1);
  if (spec == nullptr) {
    LOG_WRN("%s: failed to initialize common_speculative draft-mtp state\n",
            __func__);
    llama_free(ctx_dft);
    return nullptr;
  }

  auto *mtp = new llama_dart_mtp();
  mtp->ctx_tgt = ctx_tgt;
  mtp->ctx_dft = ctx_dft;
  mtp->spec = spec;
  return mtp;
}

LLAMADART_API struct llama_dart_mtp *llama_dart_mtp_init(
    struct llama_model *model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  return llama_dart_mtp_init_impl(model, ctx_tgt, ctx_params, draft_token_max,
                                  draft_token_min, min_probability,
                                  backend_sampling);
}

LLAMADART_API struct llama_dart_mtp *llama_dart_mtp_init_with_draft_model(
    struct llama_model *draft_model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  return llama_dart_mtp_init_impl(draft_model, ctx_tgt, ctx_params,
                                  draft_token_max, draft_token_min,
                                  min_probability, backend_sampling);
}

LLAMADART_API void llama_dart_mtp_free(struct llama_dart_mtp *mtp) {
  if (mtp == nullptr) {
    return;
  }

  if (mtp->spec != nullptr) {
    common_speculative_free(mtp->spec);
    mtp->spec = nullptr;
  }
  if (mtp->ctx_tgt != nullptr) {
    llama_set_embeddings_nextn(mtp->ctx_tgt, false, false);
  }
  if (mtp->ctx_dft != nullptr) {
    llama_free(mtp->ctx_dft);
    mtp->ctx_dft = nullptr;
  }
  delete mtp;
}

LLAMADART_API struct llama_context *
llama_dart_mtp_get_draft_context(struct llama_dart_mtp *mtp) {
  if (mtp == nullptr) {
    return nullptr;
  }
  return mtp->ctx_dft;
}

LLAMADART_API bool llama_dart_mtp_begin(struct llama_dart_mtp *mtp,
                                        llama_seq_id seq_id,
                                        const llama_token *prompt,
                                        int32_t prompt_count) {
  if (mtp == nullptr || mtp->spec == nullptr || prompt_count < 0) {
    return false;
  }

  mtp->prompt.clear();
  if (prompt != nullptr && prompt_count > 0) {
    mtp->prompt.assign(prompt, prompt + prompt_count);
  }

  common_speculative_begin(mtp->spec, seq_id, mtp->prompt);
  return true;
}

LLAMADART_API bool
llama_dart_mtp_process_batch(struct llama_dart_mtp *mtp,
                             struct llama_batch batch) {
  if (mtp == nullptr || mtp->spec == nullptr) {
    return false;
  }
  return common_speculative_process(mtp->spec, batch);
}

LLAMADART_API int32_t llama_dart_mtp_draft(
    struct llama_dart_mtp *mtp, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  if (mtp == nullptr || mtp->spec == nullptr || out_tokens == nullptr ||
      out_capacity < 0 || prompt_count < 0 || draft_token_max <= 0) {
    return -1;
  }

  mtp->prompt.clear();
  if (prompt != nullptr && prompt_count > 0) {
    mtp->prompt.assign(prompt, prompt + prompt_count);
  }
  mtp->draft.clear();
  mtp->draft.reserve(static_cast<size_t>(draft_token_max));

  common_speculative_get_draft_params(mtp->spec, seq_id) = {
      /* .drafting = */ true,
      /* .n_max    = */ draft_token_max,
      /* .n_past   = */ n_past,
      /* .id_last  = */ id_last,
      /* .prompt   = */ &mtp->prompt,
      /* .result   = */ &mtp->draft,
  };

  common_speculative_draft(mtp->spec);

  const int32_t count = std::min<int32_t>(
      static_cast<int32_t>(mtp->draft.size()), out_capacity);
  for (int32_t i = 0; i < count; ++i) {
    out_tokens[i] = mtp->draft[static_cast<size_t>(i)];
  }
  return count;
}

LLAMADART_API void llama_dart_mtp_accept(struct llama_dart_mtp *mtp,
                                         llama_seq_id seq_id,
                                         uint16_t accepted_count) {
  if (mtp == nullptr || mtp->spec == nullptr) {
    return;
  }
  common_speculative_accept(mtp->spec, seq_id, accepted_count);
}

LLAMADART_API struct llama_dart_ngram *
llama_dart_ngram_simple_init(int32_t ngram_size, int32_t draft_token_max) {
  common_params_speculative params;
  params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE};
  params.ngram_simple.size_n = llama_dart_uint16_or_default(ngram_size, 12);
  params.ngram_simple.size_m =
      llama_dart_uint16_or_default(draft_token_max, 48);

  common_speculative *spec = common_speculative_init(params, 1);
  if (spec == nullptr) {
    LOG_WRN("%s: failed to initialize common_speculative ngram-simple state\n",
            __func__);
    return nullptr;
  }

  auto *ngram = new llama_dart_ngram();
  ngram->spec = spec;
  return ngram;
}

LLAMADART_API void llama_dart_ngram_free(struct llama_dart_ngram *ngram) {
  if (ngram == nullptr) {
    return;
  }

  if (ngram->spec != nullptr) {
    common_speculative_free(ngram->spec);
    ngram->spec = nullptr;
  }
  delete ngram;
}

LLAMADART_API bool llama_dart_ngram_begin(struct llama_dart_ngram *ngram,
                                          llama_seq_id seq_id,
                                          const llama_token *prompt,
                                          int32_t prompt_count) {
  if (ngram == nullptr || ngram->spec == nullptr || prompt_count < 0) {
    return false;
  }
  if (seq_id != 0) {
    return false;
  }

  ngram->prompt.clear();
  ngram->has_last_draft = false;
  if (prompt != nullptr && prompt_count > 0) {
    ngram->prompt.assign(prompt, prompt + prompt_count);
  }

  common_speculative_begin(ngram->spec, seq_id, ngram->prompt);
  return true;
}

LLAMADART_API bool
llama_dart_ngram_process_batch(struct llama_dart_ngram *ngram,
                               struct llama_batch batch) {
  if (ngram == nullptr || ngram->spec == nullptr) {
    return false;
  }
  return common_speculative_process(ngram->spec, batch);
}

LLAMADART_API int32_t llama_dart_ngram_draft(
    struct llama_dart_ngram *ngram, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  if (ngram == nullptr || ngram->spec == nullptr || out_tokens == nullptr ||
      out_capacity < 0 || prompt_count < 0 || draft_token_max <= 0) {
    return -1;
  }
  if (seq_id != 0) {
    ngram->has_last_draft = false;
    return -1;
  }

  ngram->prompt.clear();
  if (prompt != nullptr && prompt_count > 0) {
    ngram->prompt.assign(prompt, prompt + prompt_count);
  }
  ngram->draft.clear();
  ngram->has_last_draft = false;
  ngram->draft.reserve(static_cast<size_t>(draft_token_max));

  common_speculative_get_draft_params(ngram->spec, seq_id) = {
      /* .drafting = */ true,
      /* .n_max    = */ draft_token_max,
      /* .n_past   = */ n_past,
      /* .id_last  = */ id_last,
      /* .prompt   = */ &ngram->prompt,
      /* .result   = */ &ngram->draft,
  };

  common_speculative_draft(ngram->spec);

  const int32_t count = std::min<int32_t>(
      static_cast<int32_t>(ngram->draft.size()),
      std::min(draft_token_max, out_capacity));
  ngram->has_last_draft = count > 0;
  for (int32_t i = 0; i < count; ++i) {
    out_tokens[i] = ngram->draft[static_cast<size_t>(i)];
  }
  return count;
}

LLAMADART_API void llama_dart_ngram_accept(struct llama_dart_ngram *ngram,
                                           llama_seq_id seq_id,
                                           uint16_t accepted_count) {
  if (ngram == nullptr || ngram->spec == nullptr) {
    return;
  }
  if (seq_id != 0 || !ngram->has_last_draft) {
    return;
  }
  common_speculative_accept(ngram->spec, seq_id, accepted_count);
  ngram->has_last_draft = false;
}

LLAMADART_API int32_t llama_dart_sampler_sample_and_accept_n(
    struct llama_sampler *sampler, struct llama_context *ctx,
    const int32_t *idxs, int32_t idx_count, const llama_token *draft_tokens,
    int32_t draft_count, llama_token *out_tokens, int32_t out_capacity) {
  if (sampler == nullptr || ctx == nullptr || idxs == nullptr ||
      draft_tokens == nullptr || out_tokens == nullptr || draft_count < 0 ||
      idx_count != draft_count + 1 || out_capacity < idx_count) {
    return -1;
  }

  int32_t count = 0;
  int32_t i = 0;
  const auto sample_and_accept = [sampler, ctx](int32_t idx) {
    // llama_sampler_sample accepts CPU-sampled tokens itself, but returns
    // backend-preselected tokens before that accept path.
    const bool backend_sampled =
        llama_get_sampled_token_ith(ctx, idx) != LLAMA_TOKEN_NULL;
    const llama_token id = llama_sampler_sample(sampler, ctx, idx);
    if (backend_sampled) {
      llama_sampler_accept(sampler, id);
    }
    return id;
  };

  for (; i < draft_count; ++i) {
    const llama_token id = sample_and_accept(idxs[i]);
    out_tokens[count++] = id;
    if (draft_tokens[i] != id) {
      break;
    }
  }

  if (i == draft_count) {
    const llama_token id = sample_and_accept(idxs[i]);
    out_tokens[count++] = id;
  }

  return count;
}
}
