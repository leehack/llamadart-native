#include "llama_dart_wrapper.h"

#include "common.h"
#include "llama-ext.h"
#include "log.h"
#include "reasoning-budget.h"
#include "speculative.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <limits>
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
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
  std::vector<int8_t> process_output_mask;
  bool caps_draft_process_outputs = false;
  bool has_last_draft = false;
};

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
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
}

static bool llama_dart_type_mask_has_non_mtp_draft_context(uint32_t type_mask) {
  return llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) ||
         llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH);
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
  return count;
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
  auto *budget_sampler = common_reasoning_budget_init(
      vocab, start_tokens, end_tokens, forced_tokens, budget_tokens,
      initial_state);
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

  const auto types = llama_dart_speculative_types_from_params(params_input);
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
    context_params.n_outputs_max = 1;
    context_params.embeddings = false;
    context_params.ctx_other = target_context;

    ctx_dft = llama_init_from_model(resolved_draft_model, context_params);
    if (ctx_dft == nullptr) {
      LOG_WRN("%s: failed to create speculative draft context\n", __func__);
      return nullptr;
    }
  }

  common_params_speculative params =
      llama_dart_build_speculative_params(params_input, target_context, ctx_dft);

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
    return nullptr;
  }

  auto *speculative = new llama_dart_speculative();
  speculative->ctx_tgt = target_context;
  speculative->ctx_dft = ctx_dft;
  speculative->spec = spec;
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
    llama_set_embeddings(speculative->ctx_tgt, false);
    llama_set_embeddings_nextn(speculative->ctx_tgt, false, false);
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
  return speculative != nullptr && speculative->spec != nullptr &&
         common_speculative_need_embd(speculative->spec);
}

LLAMADART_API bool llama_dart_speculative_need_embd_nextn(
    struct llama_dart_speculative *speculative) {
  return speculative != nullptr && speculative->spec != nullptr &&
         common_speculative_need_embd_nextn(speculative->spec);
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
