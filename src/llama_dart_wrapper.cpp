#include "llama_dart_wrapper.h"

#include "common.h"
#include "llama-ext.h"
#include "log.h"
#include "speculative.h"

#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <string>
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
  for (; i < draft_count; ++i) {
    const llama_token id = llama_sampler_sample(sampler, ctx, idxs[i]);
    out_tokens[count++] = id;
    if (draft_tokens[i] != id) {
      break;
    }
  }

  if (i == draft_count) {
    const llama_token id = llama_sampler_sample(sampler, ctx, idxs[i]);
    out_tokens[count++] = id;
  }

  return count;
}
}
