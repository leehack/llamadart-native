#include "llama_dart_wrapper.h"
#include "llama_dart_mtp_internal.h"
#include "llama_dart_mtmd_compat.h"
#include "llama_dart_speculative_compat.h"
#include "llama_dart_tts_eval_internal.h"

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
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#include <dlfcn.h>
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

// The calling thread's last caught exception. It has no destructor and needs
// no allocation, so recording std::bad_alloc cannot fail and nothing is left
// to destroy when the thread or the process ends.
struct llama_dart_error_state {
  bool set;
  char message[512];
};

static thread_local llama_dart_error_state llama_dart_error = {};

static void llama_dart_error_record(const char *message) noexcept {
  snprintf(llama_dart_error.message, sizeof(llama_dart_error.message), "%s",
           message != nullptr ? message : "unknown C++ exception");
  llama_dart_error.set = true;
}

// Runs call so that no C++ exception leaves libllamadart: an exception unwinds
// call, is recorded as the thread's last error, and failure is returned.
template <typename Result, typename Call>
static Result llama_dart_catch(Result failure, Call &&call) noexcept {
  try {
    return call();
  } catch (const std::exception &error) {
    llama_dart_error_record(error.what());
  } catch (...) {
    llama_dart_error_record(nullptr);
  }
  return failure;
}

// The same for a call whose caller reads the last error afterwards: it starts
// with none, so that a last error after the call is this call's.
template <typename Result, typename Call>
static Result llama_dart_barrier(Result failure, Call &&call) noexcept {
  llama_dart_error.set = false;
  return llama_dart_catch(failure, call);
}

template <typename Call>
static void llama_dart_void_barrier(Call &&call) noexcept {
  llama_dart_barrier(false, [&call] {
    call();
    return true;
  });
}

// For the functions that free an object. A Dart finalizer may run one on a
// thread between a call that failed there and the read of its error, so they
// leave the last error alone unless they catch an exception themselves.
template <typename Call>
static void llama_dart_free_barrier(Call &&call) noexcept {
  llama_dart_catch(false, [&call] {
    call();
    return true;
  });
}

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
  const int8_t *cancel_flag = nullptr;
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

static bool llama_dart_tts_cancelled(llama_dart_tts *tts) {
  return llama_dart_tts_cancel_observed(&tts->cancel_requested,
                                        tts->cancel_flag);
}

struct llama_dart_tts_eval_scope;

static thread_local llama_dart_tts_eval_scope *llama_dart_tts_active_eval =
    nullptr;

struct llama_dart_tts_eval_scope {
  llama_dart_tts *tts;
  llama_dart_tts_eval_chunker chunker;
  llama_dart_tts_eval_scope *previous;

  explicit llama_dart_tts_eval_scope(llama_dart_tts *task)
      : tts(task), previous(llama_dart_tts_active_eval) {
    llama_dart_tts_active_eval = this;
  }
  ~llama_dart_tts_eval_scope() { llama_dart_tts_active_eval = previous; }
  llama_dart_tts_eval_scope(const llama_dart_tts_eval_scope &) = delete;
  llama_dart_tts_eval_scope &
  operator=(const llama_dart_tts_eval_scope &) = delete;
};

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
  tts->cancel_flag = nullptr;
}

// A TTS call behind the barrier. An exception fails the task like any other
// upstream error, so that the next step does not continue it.
template <typename Call>
static llama_dart_tts_status llama_dart_tts_barrier(llama_dart_tts *tts,
                                                    Call &&call) noexcept {
  const llama_dart_tts_status status =
      llama_dart_barrier(LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR, call);
  if (tts != nullptr && llama_dart_error.set) {
    // Releasing the task's resources calls llama.cpp again.
    llama_dart_exit_call_begin();
    try {
      llama_dart_tts_fail(tts, LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR,
                          llama_dart_error.message);
    } catch (...) {
      tts->state = LLAMA_DART_TTS_STATE_FAILED;
    }
    llama_dart_exit_call_end();
  }
  return status;
}

static llama_dart_tts_status llama_dart_tts_mark_cancelled(
    llama_dart_tts *tts) {
  tts->state = LLAMA_DART_TTS_STATE_CANCELLED;
  tts->error = "TTS task cancelled";
  llama_dart_tts_release_task_resources(tts);
  return LLAMA_DART_TTS_STATUS_CANCELLED;
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
  const int32_t output_status = mtmd_helper_gen_audio_get_output(
      tts->generator, &sample_rate, &data, &data_len, &sample_count);
  if (llama_dart_tts_cancelled(tts)) {
    return llama_dart_tts_mark_cancelled(tts);
  }
  if (output_status != 0) {
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

struct llama_dart_exit_entry {
  void (*free_fn)(void *) = nullptr;
  int32_t stage = 0;
  uint64_t order = 0;
  // Objects the entry uses without owning them. Teardown frees the entry only
  // while they are tracked: a caller that creates them with upstream functions
  // may be using them in calls teardown cannot see.
  void *uses[2] = {nullptr, nullptr};
};

// A thread that just left a call in flight usually still makes short calls on
// the same objects, such as sampling after a decode, before its next call in
// flight blocks it. Teardown gives it this long to get there.
static const std::chrono::milliseconds llama_dart_exit_settle_time{250};

struct llama_dart_exit_registry {
  std::mutex mutex;
  std::condition_variable idle;
  std::unordered_map<void *, llama_dart_exit_entry> objects;
  std::chrono::steady_clock::time_point last_call_end{};
  uint64_t next_order = 0;
  int32_t calls = 0;
  int32_t creating_calls = 0;
  // Frees in flight. The object of one is no longer in objects, and exit must
  // not go on to destroy the statics that the free still uses.
  int32_t freeing_calls = 0;
  int32_t wait_ms = 2000;
  std::atomic<bool> armed{false};
  std::atomic<bool> torn_down{false};
};

static llama_dart_exit_registry &llama_dart_exit_state() {
  // Never destroyed: other threads still reach it after static destructors.
  static auto *registry = new llama_dart_exit_registry();
  return *registry;
}

static thread_local bool llama_dart_exit_teardown_thread = false;

// Calls in flight on this thread. The registry counts the outermost one.
static thread_local int32_t llama_dart_exit_call_depth = 0;

// Called with the registry locked. Returns whether the calling thread may go
// on to use tracked objects. Once teardown has begun, teardown waits for a
// thread that is in a call in flight, and the teardown thread gets false. Any
// other thread may hold objects that teardown frees, so it never returns to
// its caller.
static bool llama_dart_exit_admit(llama_dart_exit_registry &registry,
                                  std::unique_lock<std::mutex> &lock) {
  if (!registry.torn_down) {
    return true;
  }
  if (llama_dart_exit_teardown_thread) {
    return false;
  }
  if (llama_dart_exit_call_depth > 0) {
    return true;
  }
  lock.unlock();
  for (;;) {
    std::this_thread::sleep_for(std::chrono::hours(1));
  }
}

#if defined(__APPLE__)
struct llama_dart_exit_static_destructor {
  void (*destroy)(void *);
  void *object;
};

static void llama_dart_exit_destroy_static(void *argument) {
  const auto destructor =
      *static_cast<llama_dart_exit_static_destructor *>(argument);
  free(argument);
  llama_dart_exit_teardown();
  destructor.destroy(destructor.object);
}

using llama_dart_exit_register_destructor = int (*)(void (*)(void *), void *,
                                                    void *);

static std::atomic<llama_dart_exit_register_destructor>
    llama_dart_exit_system_cxa_atexit{nullptr};

extern "C" {
// The C++ runtime registers the destructor of every static in this image
// through __cxa_atexit, and the linker binds those calls to this definition.
// Tracked objects use such statics, some of which llama.cpp creates on first
// use at any time, so teardown has to run before the first of them is
// destroyed: each destructor is registered behind a call to teardown. A plain
// atexit handler cannot do that, as it only precedes the statics that exist
// when it is registered, and registering it again on every use grows the
// handler list without bound.
__attribute__((visibility("hidden"))) int
__cxa_atexit(void (*destroy)(void *), void *object, void *dso_handle) {
  auto system_cxa_atexit = llama_dart_exit_system_cxa_atexit.load();
  if (system_cxa_atexit == nullptr) {
    system_cxa_atexit = reinterpret_cast<llama_dart_exit_register_destructor>(
        dlsym(RTLD_NEXT, "__cxa_atexit"));
    if (system_cxa_atexit == nullptr) {
      return -1;
    }
    llama_dart_exit_system_cxa_atexit.store(system_cxa_atexit);
  }
  auto *destructor = static_cast<llama_dart_exit_static_destructor *>(
      malloc(sizeof(llama_dart_exit_static_destructor)));
  if (destructor == nullptr) {
    return -1;
  }
  *destructor = {destroy, object};
  const int status = system_cxa_atexit(llama_dart_exit_destroy_static,
                                       destructor, dso_handle);
  if (status != 0) {
    free(destructor);
  }
  return status;
}
}
#endif

// Registers teardown to run at exit once something is tracked, so that it
// does not wait for the first static of this image to be destroyed. atexit
// handlers run in reverse order of registration, so this one runs after the
// destructors of statics created later; __cxa_atexit above covers those.
static void llama_dart_exit_arm() {
#if defined(__APPLE__)
  if (!llama_dart_exit_state().armed.exchange(true)) {
    atexit(llama_dart_exit_teardown);
  }
#endif
}

static bool llama_dart_exit_insert(void *object, void (*free_fn)(void *),
                                   int32_t stage, void *uses_first = nullptr,
                                   void *uses_second = nullptr) {
  if (object == nullptr || free_fn == nullptr ||
      stage < LLAMA_DART_EXIT_STAGE_SESSION ||
      stage > LLAMA_DART_EXIT_STAGE_MODEL) {
    return false;
  }
  auto &registry = llama_dart_exit_state();
  {
    std::unique_lock<std::mutex> lock(registry.mutex);
    if (!llama_dart_exit_admit(registry, lock)) {
      return false;
    }
    registry.objects[object] = {
        free_fn, stage, registry.next_order++, {uses_first, uses_second}};
  }
  llama_dart_exit_arm();
  return true;
}

// Called with the registry locked.
static bool llama_dart_exit_can_free(const llama_dart_exit_registry &registry,
                                     const llama_dart_exit_entry &entry) {
  for (void *used : entry.uses) {
    if (used != nullptr && registry.objects.count(used) == 0) {
      return false;
    }
  }
  return true;
}

// Frees object once: with the function it was tracked with, or with
// untracked_free when it is not tracked.
static void llama_dart_exit_release(void *object,
                                    void (*untracked_free)(void *)) {
  if (object == nullptr) {
    return;
  }
  auto &registry = llama_dart_exit_state();
  void (*free_fn)(void *) = untracked_free;
  {
    std::unique_lock<std::mutex> lock(registry.mutex);
    if (!llama_dart_exit_admit(registry, lock)) {
      return;
    }
    const auto found = registry.objects.find(object);
    if (found != registry.objects.end()) {
      free_fn = found->second.free_fn;
      registry.objects.erase(found);
    }
    if (free_fn == nullptr) {
      return;
    }
    if (llama_dart_exit_call_depth++ == 0) {
      ++registry.calls;
    }
    // Counted under the lock that untracked the object, so that teardown sees
    // either the object or its free.
    ++registry.freeing_calls;
  }
  // Ends the free also when free_fn throws.
  struct freeing_call {
    llama_dart_exit_registry &registry;
    ~freeing_call() {
      {
        std::lock_guard<std::mutex> lock(registry.mutex);
        --registry.freeing_calls;
      }
      llama_dart_exit_call_end();
    }
  } freeing{registry};
  free_fn(object);
}

// Begins a call in flight on this thread. A creating call is counted as one
// under the same lock, so that teardown cannot run in between and return
// before the creation starts.
static void llama_dart_exit_begin_call(bool creating) {
  if (!creating && llama_dart_exit_call_depth > 0) {
    ++llama_dart_exit_call_depth;
    return;
  }
  auto &registry = llama_dart_exit_state();
  std::unique_lock<std::mutex> lock(registry.mutex);
  if (llama_dart_exit_call_depth > 0) {
    ++llama_dart_exit_call_depth;
  } else if (llama_dart_exit_admit(registry, lock)) {
    ++registry.calls;
    llama_dart_exit_call_depth = 1;
  }
  if (creating) {
    ++registry.creating_calls;
  }
}

struct llama_dart_exit_call {
  llama_dart_exit_call() { llama_dart_exit_call_begin(); }
  ~llama_dart_exit_call() { llama_dart_exit_call_end(); }
  llama_dart_exit_call(const llama_dart_exit_call &) = delete;
  llama_dart_exit_call &operator=(const llama_dart_exit_call &) = delete;
};

// A call in flight that tracks what it creates. Teardown waits for it even
// when nothing is tracked yet.
struct llama_dart_exit_creating_call {
  llama_dart_exit_creating_call() { llama_dart_exit_begin_call(true); }
  ~llama_dart_exit_creating_call() {
    {
      auto &registry = llama_dart_exit_state();
      std::lock_guard<std::mutex> lock(registry.mutex);
      --registry.creating_calls;
    }
    llama_dart_exit_call_end();
  }
  llama_dart_exit_creating_call(const llama_dart_exit_creating_call &) =
      delete;
  llama_dart_exit_creating_call &
  operator=(const llama_dart_exit_creating_call &) = delete;
};

// Runs call, which decodes or encodes on ctx, as a call in flight behind the
// barrier. A backend may still be computing when llama_decode or llama_encode
// returns, and the next read of the context then waits for it. That wait must
// be part of the call in flight, or teardown frees the context under it, so
// the call in flight ends only after it, also when call throws.
template <typename Call>
static int32_t llama_dart_exit_evaluate(llama_context *ctx, Call &&call) {
  return llama_dart_barrier<int32_t>(
      LLAMA_DART_STATUS_EXCEPTION, [ctx, &call]() -> int32_t {
        llama_dart_exit_call in_flight;
        int32_t status;
        try {
          status = call();
        } catch (...) {
          // The exception of the call is the one to report.
          try {
            llama_synchronize(ctx);
          } catch (...) {
          }
          throw;
        }
        llama_synchronize(ctx);
        return status;
      });
}

// Tracks an object that a creating call just made and returns it. An object
// that cannot be tracked is freed, so that an exception leaves nothing behind
// that exit teardown does not know.
template <typename Object>
static Object *llama_dart_exit_track_created(Object *object,
                                             void (*free_fn)(void *),
                                             int32_t stage,
                                             void *uses_first = nullptr,
                                             void *uses_second = nullptr) {
  try {
    llama_dart_exit_insert(object, free_fn, stage, uses_first, uses_second);
  } catch (...) {
    if (object != nullptr) {
      free_fn(object);
    }
    throw;
  }
  return object;
}

struct llama_dart_exit_load_progress {
  llama_progress_callback callback;
  void *user_data;
};

static bool llama_dart_exit_load_progress_callback(float progress,
                                                   void *user_data) {
  if (llama_dart_exit_state().torn_down.load(std::memory_order_relaxed)) {
    return false;
  }
  const auto *caller =
      static_cast<const llama_dart_exit_load_progress *>(user_data);
  return caller->callback == nullptr ||
         caller->callback(progress, caller->user_data);
}

static void llama_dart_exit_free_model(void *object) {
  llama_model_free(static_cast<llama_model *>(object));
}

static void llama_dart_exit_free_context(void *object) {
  llama_free(static_cast<llama_context *>(object));
}

static void llama_dart_exit_free_mtmd(void *object) {
  mtmd_free(static_cast<mtmd_context *>(object));
}

extern "C" {

static void llama_dart_tts_free_object(void *object);
static void llama_dart_speculative_free_object(void *object);
static void llama_dart_mtp_free_object(void *object);
static void llama_dart_ngram_free_object(void *object);

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

LLAMADART_API const char *llama_dart_last_error(void) {
  return llama_dart_error.set ? llama_dart_error.message : nullptr;
}

LLAMADART_API void llama_dart_clear_last_error(void) {
  llama_dart_error.set = false;
}

LLAMADART_API bool llama_dart_exit_track(void *object, void (*free_fn)(void *),
                                        int32_t stage) {
  return llama_dart_barrier(false, [object, free_fn, stage] {
    return llama_dart_exit_insert(object, free_fn, stage);
  });
}

LLAMADART_API bool llama_dart_exit_untrack(void *object) {
  auto &registry = llama_dart_exit_state();
  std::unique_lock<std::mutex> lock(registry.mutex);
  if (!llama_dart_exit_admit(registry, lock)) {
    return false;
  }
  return registry.objects.erase(object) != 0;
}

LLAMADART_API void llama_dart_exit_free(void *object) {
  llama_dart_free_barrier(
      [object] { llama_dart_exit_release(object, nullptr); });
}

LLAMADART_API int32_t llama_dart_exit_tracked_count(void) {
  auto &registry = llama_dart_exit_state();
  std::lock_guard<std::mutex> lock(registry.mutex);
  return static_cast<int32_t>(registry.objects.size());
}

LLAMADART_API void llama_dart_exit_call_begin(void) {
  llama_dart_exit_begin_call(false);
}

LLAMADART_API void llama_dart_exit_call_end(void) {
  if (llama_dart_exit_call_depth == 0 || --llama_dart_exit_call_depth > 0) {
    return;
  }
  auto &registry = llama_dart_exit_state();
  std::unique_lock<std::mutex> lock(registry.mutex);
  --registry.calls;
  registry.idle.notify_all();
  if (llama_dart_exit_admit(registry, lock)) {
    registry.last_call_end = std::chrono::steady_clock::now();
  }
}

LLAMADART_API void llama_dart_exit_set_wait_ms(int32_t wait_ms) {
  auto &registry = llama_dart_exit_state();
  std::lock_guard<std::mutex> lock(registry.mutex);
  registry.wait_ms = std::max(wait_ms, 0);
}

LLAMADART_API void llama_dart_exit_teardown(void) {
  auto &registry = llama_dart_exit_state();
  std::vector<std::pair<void *, llama_dart_exit_entry>> objects;
  {
    std::unique_lock<std::mutex> lock(registry.mutex);
    if (registry.torn_down.exchange(true)) {
      return;
    }
    llama_dart_exit_teardown_thread = true;
    const auto can_free = [&registry](const auto &tracked) {
      return llama_dart_exit_can_free(registry, tracked.second);
    };
    // With nothing to free and nothing being created or freed, a call in
    // flight is no reason to hold up the exit.
    if (registry.creating_calls == 0 && registry.freeing_calls == 0 &&
        std::none_of(registry.objects.begin(), registry.objects.end(),
                     can_free)) {
      return;
    }
    // A call in flight on this thread cannot end while teardown runs.
    const int32_t own_calls = llama_dart_exit_call_depth > 0 ? 1 : 0;
    const bool idle = registry.idle.wait_for(
        lock, std::chrono::milliseconds(registry.wait_ms),
        [&registry, own_calls] { return registry.calls == own_calls; });
    if (!idle) {
      return;
    }
    registry.idle.wait_until(
        lock, registry.last_call_end + llama_dart_exit_settle_time,
        [] { return false; });
    std::copy_if(registry.objects.begin(), registry.objects.end(),
                 std::back_inserter(objects), can_free);
  }
  std::sort(objects.begin(), objects.end(), [](const auto &a, const auto &b) {
    return a.second.stage != b.second.stage
               ? a.second.stage < b.second.stage
               : a.second.order > b.second.order;
  });
  for (const auto &[object, entry] : objects) {
    {
      std::lock_guard<std::mutex> lock(registry.mutex);
      registry.objects.erase(object);
    }
    // An exception here would end the process in std::terminate, the abort
    // at exit that teardown exists to prevent.
    try {
      entry.free_fn(object);
    } catch (...) {
    }
  }
}

LLAMADART_API struct llama_model *
llama_dart_model_load_from_file(const char *path_model,
                                struct llama_model_params params) {
  return llama_dart_barrier<llama_model *>(nullptr, [path_model, &params] {
    llama_dart_exit_creating_call call;
    llama_dart_exit_load_progress progress{params.progress_callback,
                                           params.progress_callback_user_data};
    params.progress_callback = llama_dart_exit_load_progress_callback;
    params.progress_callback_user_data = &progress;
    return llama_dart_exit_track_created(
        llama_model_load_from_file(path_model, params),
        llama_dart_exit_free_model, LLAMA_DART_EXIT_STAGE_MODEL);
  });
}

LLAMADART_API struct llama_context *
llama_dart_init_from_model(struct llama_model *model,
                           struct llama_context_params params) {
  return llama_dart_barrier<llama_context *>(nullptr, [model, &params] {
    llama_dart_exit_creating_call call;
    return llama_dart_exit_track_created(
        llama_init_from_model(model, params), llama_dart_exit_free_context,
        LLAMA_DART_EXIT_STAGE_CONTEXT);
  });
}

LLAMADART_API struct mtmd_context *llama_dart_mtmd_init_from_file(
    const char *mmproj_fname, const struct llama_model *text_model,
    const struct mtmd_context_params *ctx_params) {
  if (ctx_params == nullptr) {
    return nullptr;
  }
  return llama_dart_barrier<mtmd_context *>(
      nullptr, [mmproj_fname, text_model, ctx_params] {
        llama_dart_exit_creating_call call;
        return llama_dart_exit_track_created(
            mtmd_init_from_file(mmproj_fname, text_model, *ctx_params),
            llama_dart_exit_free_mtmd, LLAMA_DART_EXIT_STAGE_MODEL_USER);
      });
}

LLAMADART_API int32_t llama_dart_decode(struct llama_context *ctx,
                                        struct llama_batch batch) {
  return llama_dart_exit_evaluate(
      ctx, [ctx, &batch] { return llama_decode(ctx, batch); });
}

LLAMADART_API int32_t llama_dart_encode(struct llama_context *ctx,
                                        struct llama_batch batch) {
  return llama_dart_exit_evaluate(
      ctx, [ctx, &batch] { return llama_encode(ctx, batch); });
}

LLAMADART_API void llama_dart_synchronize(struct llama_context *ctx) {
  llama_dart_void_barrier([ctx] {
    llama_dart_exit_call call;
    llama_synchronize(ctx);
  });
}

LLAMADART_API llama_token llama_dart_sampler_sample(struct llama_sampler *smpl,
                                                    struct llama_context *ctx,
                                                    int32_t idx) {
  return llama_dart_barrier<llama_token>(LLAMA_TOKEN_NULL, [smpl, ctx, idx] {
    llama_dart_exit_call call;
    return llama_sampler_sample(smpl, ctx, idx);
  });
}

LLAMADART_API bool llama_dart_sampler_accept(struct llama_sampler *smpl,
                                             llama_token token) {
  return llama_dart_barrier(false, [smpl, token] {
    llama_dart_exit_call call;
    llama_sampler_accept(smpl, token);
    return true;
  });
}

LLAMADART_API bool llama_dart_state_save_file(struct llama_context *ctx,
                                              const char *path_session,
                                              const llama_token *tokens,
                                              size_t n_token_count) {
  return llama_dart_barrier(false, [=] {
    llama_dart_exit_call call;
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
  });
}

LLAMADART_API bool llama_dart_state_load_file(struct llama_context *ctx,
                                              const char *path_session,
                                              llama_token *tokens_out,
                                              size_t n_token_capacity,
                                              size_t *n_token_count_out) {
  return llama_dart_barrier(false, [=] {
    llama_dart_exit_call call;
    return llama_state_load_file(ctx, path_session, tokens_out,
                                 n_token_capacity, n_token_count_out);
  });
}

LLAMADART_API size_t llama_dart_state_seq_get_size_ext(
    struct llama_context *ctx, llama_seq_id seq_id, uint32_t flags) {
  return llama_dart_barrier<size_t>(0, [=] {
    llama_dart_exit_call call;
    return llama_state_seq_get_size_ext(ctx, seq_id, flags);
  });
}

LLAMADART_API size_t llama_dart_state_seq_get_data_ext(
    struct llama_context *ctx, uint8_t *dst, size_t size, llama_seq_id seq_id,
    uint32_t flags) {
  return llama_dart_barrier<size_t>(0, [=] {
    llama_dart_exit_call call;
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags);
  });
}

LLAMADART_API size_t llama_dart_state_seq_set_data_ext(
    struct llama_context *ctx, const uint8_t *src, size_t size,
    llama_seq_id dest_seq_id, uint32_t flags) {
  return llama_dart_barrier<size_t>(0, [=] {
    llama_dart_exit_call call;
    return llama_state_seq_set_data_ext(ctx, src, size, dest_seq_id, flags);
  });
}

LLAMADART_API struct llama_adapter_lora *
llama_dart_adapter_lora_init(struct llama_model *model,
                             const char *path_lora) {
  return llama_dart_barrier<llama_adapter_lora *>(nullptr, [=] {
    llama_dart_exit_call call;
    return llama_adapter_lora_init(model, path_lora);
  });
}

LLAMADART_API int32_t llama_dart_mtmd_tokenize(
    const struct mtmd_context *ctx, struct mtmd_input_chunks *output,
    const struct mtmd_input_text *text,
    const struct mtmd_bitmap *const *bitmaps, size_t n_bitmaps) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    llama_dart_exit_call call;
    return mtmd_tokenize(ctx, output, text, bitmaps, n_bitmaps);
  });
}

LLAMADART_API int32_t
llama_dart_mtmd_encode_chunk(struct mtmd_context *ctx,
                             const struct mtmd_input_chunk *chunk) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    llama_dart_exit_call call;
    return mtmd_encode_chunk(ctx, chunk);
  });
}

LLAMADART_API int32_t llama_dart_mtmd_helper_eval_chunks(
    struct mtmd_context *ctx, struct llama_context *lctx,
    const struct mtmd_input_chunks *chunks, llama_pos n_past,
    llama_seq_id seq_id, int32_t n_batch, bool logits_last,
    llama_pos *new_n_past) {
  return llama_dart_exit_evaluate(lctx, [=] {
    return mtmd_helper_eval_chunks(ctx, lctx, chunks, n_past, seq_id, n_batch,
                                   logits_last, new_n_past);
  });
}

LLAMADART_API int32_t llama_dart_mtmd_helper_eval_chunk_single(
    struct mtmd_context *ctx, struct llama_context *lctx,
    const struct mtmd_input_chunk *chunk, llama_pos n_past,
    llama_seq_id seq_id, int32_t n_batch, bool logits_last,
    llama_pos *new_n_past) {
  return llama_dart_exit_evaluate(lctx, [=] {
    return mtmd_helper_eval_chunk_single(ctx, lctx, chunk, n_past, seq_id,
                                         n_batch, logits_last, new_n_past);
  });
}

LLAMADART_API int32_t llama_dart_mtmd_helper_decode_image_chunk(
    struct mtmd_context *ctx, struct llama_context *lctx,
    const struct mtmd_input_chunk *chunk, float *encoded_embd,
    llama_pos n_past, llama_seq_id seq_id, int32_t n_batch,
    llama_pos *new_n_past,
    int32_t (*callback)(struct llama_batch batch, void *user_data),
    void *user_data) {
  return llama_dart_exit_evaluate(lctx, [=] {
    return llama_dart_decode_image_chunk_compat(
        ctx, lctx, chunk, encoded_embd, n_past, seq_id, n_batch, new_n_past,
        callback, user_data);
  });
}

LLAMADART_API enum ggml_status
llama_dart_ggml_backend_sched_graph_compute(ggml_backend_sched_t sched,
                                            struct ggml_cgraph *graph) {
  return llama_dart_barrier(GGML_STATUS_FAILED, [sched, graph] {
    llama_dart_exit_call call;
    return ggml_backend_sched_graph_compute(sched, graph);
  });
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

static enum llama_dart_tts_status llama_dart_tts_get_info_impl(
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

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_get_info(
    const struct mtmd_context *mtmd, struct llama_dart_tts_info *out_info) {
  return llama_dart_barrier(LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR, [=] {
    return llama_dart_tts_get_info_impl(mtmd, out_info);
  });
}

static struct llama_dart_tts *llama_dart_tts_init_impl(
    struct llama_context *llama, struct mtmd_context *mtmd,
    enum llama_dart_tts_status *out_status) {
  llama_dart_exit_creating_call call;
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
  llama_dart_tts *tts = nullptr;
  try {
    tts = new llama_dart_tts();
  } catch (...) {
    mtmd_helper_gen_audio_free(generator);
    throw;
  }
  tts->llama = llama;
  tts->mtmd = mtmd;
  tts->generator = generator;
  llama_dart_exit_track_created(tts, llama_dart_tts_free_object,
                                LLAMA_DART_EXIT_STAGE_SESSION, llama, mtmd);
  if (out_status != nullptr) {
    *out_status = LLAMA_DART_TTS_STATUS_OK;
  }
  return tts;
}

LLAMADART_API struct llama_dart_tts *llama_dart_tts_init(
    struct llama_context *llama, struct mtmd_context *mtmd,
    enum llama_dart_tts_status *out_status) {
  llama_dart_tts *tts = llama_dart_barrier<llama_dart_tts *>(nullptr, [=] {
    return llama_dart_tts_init_impl(llama, mtmd, out_status);
  });
  if (tts == nullptr && llama_dart_error.set && out_status != nullptr) {
    *out_status = LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR;
  }
  return tts;
}

LLAMADART_API void llama_dart_tts_free(struct llama_dart_tts *tts) {
  llama_dart_free_barrier(
      [tts] { llama_dart_exit_release(tts, llama_dart_tts_free_object); });
}

static void llama_dart_tts_free_object(void *object) {
  auto *tts = static_cast<llama_dart_tts *>(object);
  llama_dart_tts_release_task_resources(tts);
  mtmd_helper_gen_audio_free(tts->generator);
  tts->generator = nullptr;
  delete tts;
}

static enum llama_dart_tts_status llama_dart_tts_start_impl(
    struct llama_dart_tts *tts,
    const struct llama_dart_tts_request *request) {
  llama_dart_exit_call call;
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
    mtmd_helper_bitmap_wrapper wrapper = llama_dart_bitmap_from_buffer(
        tts->mtmd, request->speaker_audio, request->speaker_audio_length);
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

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_start(
    struct llama_dart_tts *tts,
    const struct llama_dart_tts_request *request) {
  return llama_dart_tts_barrier(
      tts, [=] { return llama_dart_tts_start_impl(tts, request); });
}

static enum llama_dart_tts_status llama_dart_tts_step_impl(
    struct llama_dart_tts *tts,
    struct llama_dart_tts_progress *out_progress) {
  llama_dart_exit_call call;
  if (tts == nullptr || out_progress == nullptr ||
      out_progress->struct_size < sizeof(*out_progress)) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  const bool active = tts->state == LLAMA_DART_TTS_STATE_PROCESSING_PROMPT ||
                      tts->state == LLAMA_DART_TTS_STATE_GENERATING;
  if (active && llama_dart_tts_cancelled(tts)) {
    const auto status = llama_dart_tts_mark_cancelled(tts);
    llama_dart_tts_write_progress(tts, out_progress);
    return status;
  }
  llama_dart_tts_eval_scope eval_scope(tts);
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
    const bool generated =
        state != nullptr &&
        llama_dart_tts_step_gen(&mtmd_helper_gen_audio_step_gen,
                                tts->generator, sampled, state, &next_state,
                                &stop) == 0;
    if (llama_dart_tts_cancelled(tts)) {
      const auto status = llama_dart_tts_mark_cancelled(tts);
      llama_dart_tts_write_progress(tts, out_progress);
      return status;
    }
    if (!generated) {
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

LLAMADART_API enum llama_dart_tts_status llama_dart_tts_step(
    struct llama_dart_tts *tts,
    struct llama_dart_tts_progress *out_progress) {
  return llama_dart_tts_barrier(
      tts, [=] { return llama_dart_tts_step_impl(tts, out_progress); });
}

LLAMADART_API void llama_dart_tts_cancel(struct llama_dart_tts *tts) {
  if (tts != nullptr) {
    tts->cancel_requested.store(true, std::memory_order_release);
  }
}

LLAMADART_API enum llama_dart_tts_status
llama_dart_tts_set_cancel_flag(struct llama_dart_tts *tts,
                               const int8_t *flag) {
  if (tts == nullptr || flag == nullptr) {
    return LLAMA_DART_TTS_STATUS_INVALID_ARGUMENT;
  }
  if (tts->state != LLAMA_DART_TTS_STATE_PROCESSING_PROMPT &&
      tts->state != LLAMA_DART_TTS_STATE_GENERATING) {
    return llama_dart_barrier(LLAMA_DART_TTS_STATUS_UPSTREAM_ERROR, [tts] {
      return llama_dart_tts_error(tts, LLAMA_DART_TTS_STATUS_INVALID_STATE,
                                  "no TTS task is active");
    });
  }
  tts->cancel_flag = flag;
  return LLAMA_DART_TTS_STATUS_OK;
}

LLAMADART_API bool llama_dart_tts_eval_callback(struct ggml_tensor *tensor,
                                                bool ask, void *user_data) {
  (void)user_data;
  llama_dart_tts_eval_scope *scope = llama_dart_tts_active_eval;
  if (scope == nullptr) {
    return !ask;
  }
  return llama_dart_tts_eval_answer(&scope->chunker, tensor, ask,
                                    llama_dart_tts_cancelled(scope->tts));
}

static enum llama_dart_tts_status
llama_dart_tts_reset_impl(struct llama_dart_tts *tts) {
  llama_dart_exit_call call;
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

LLAMADART_API enum llama_dart_tts_status
llama_dart_tts_reset(struct llama_dart_tts *tts) {
  return llama_dart_tts_barrier(
      tts, [=] { return llama_dart_tts_reset_impl(tts); });
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

static struct llama_sampler *llama_dart_sampler_init_reasoning_budget_impl(
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
  // Keep grammar_sampler owned by the caller on initialization failure, also
  // when it fails with an exception. The Dart bridge frees it when this
  // function returns nullptr; ownership moves to the composite sampler only
  // after a non-null result is returned.
  std::unique_ptr<llama_sampler, void (*)(llama_sampler *)> budget_sampler(
      llama_dart_reasoning_budget_init_compat(
          &common_reasoning_budget_init, vocab, start_tokens, end_tokens,
          forced_tokens, budget_tokens, initial_state),
      llama_sampler_free);
  if (budget_sampler == nullptr) {
    return nullptr;
  }

  std::unique_ptr<llama_dart_reasoning_budget> context(
      new llama_dart_reasoning_budget{
          /* .budget                        = */ budget_sampler.get(),
          /* .grammar                       = */ grammar_sampler,
          /* .pause_grammar_while_reasoning = */ pause_grammar_while_reasoning,
      });
  auto *result =
      llama_sampler_init(&llama_dart_reasoning_budget_interface, context.get());
  if (result != nullptr) {
    budget_sampler.release();
    context.release();
  }
  return result;
}

LLAMADART_API struct llama_sampler *llama_dart_sampler_init_reasoning_budget(
    const struct llama_vocab *vocab, const char *start_tag, const char *end_tag,
    const char *forced_message, int32_t budget_tokens,
    bool pause_grammar_while_reasoning,
    struct llama_sampler *grammar_sampler, const llama_token *prompt_tokens,
    int32_t prompt_token_count) {
  return llama_dart_barrier<llama_sampler *>(nullptr, [=] {
    return llama_dart_sampler_init_reasoning_budget_impl(
        vocab, start_tag, end_tag, forced_message, budget_tokens,
        pause_grammar_while_reasoning, grammar_sampler, prompt_tokens,
        prompt_token_count);
  });
}

static struct llama_dart_speculative *llama_dart_speculative_init_impl(
    struct llama_model *target_model, struct llama_model *draft_model,
    struct llama_context *target_context,
    struct llama_context_params context_params,
    const struct llama_dart_speculative_params *dart_params) {
  llama_dart_exit_creating_call call;
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
    throw;
  } catch (...) {
    LOG_WRN("%s: failed to resolve speculative types\n", __func__);
    throw;
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

  // Owns what is created from here on, so that neither a failure nor an
  // exception leaves a draft context behind that nothing frees.
  struct deleter {
    void operator()(llama_dart_speculative *created) const {
      llama_dart_speculative_free_object(created);
    }
  };
  std::unique_ptr<llama_dart_speculative, deleter> speculative(
      new llama_dart_speculative());

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

    speculative->ctx_dft =
        llama_init_from_model(resolved_draft_model, context_params);
    if (speculative->ctx_dft == nullptr) {
      LOG_WRN("%s: failed to create speculative draft context\n", __func__);
      return nullptr;
    }
  }

  llama_context *ctx_dft = speculative->ctx_dft;
  params.draft.ctx_dft = ctx_dft;
  speculative->target_output_layer_ids =
      llama_dart_speculative_target_output_layer_ids(
          target_model, ctx_dft == nullptr ? nullptr : llama_get_model(ctx_dft));

  // Initialization may change the target's outputs. Freeing resets them once
  // the target is set.
  speculative->ctx_tgt = target_context;
  try {
    speculative->spec = common_speculative_init(params, 1);
  } catch (const std::exception &e) {
    LOG_WRN("%s: failed to initialize common_speculative: %s\n", __func__,
            e.what());
    throw;
  } catch (...) {
    LOG_WRN("%s: failed to initialize common_speculative\n", __func__);
    throw;
  }
  if (speculative->spec == nullptr) {
    return nullptr;
  }

  speculative->embedding_requirements =
      llama_dart_speculative_embedding_requirements_for(types);
  speculative->caps_draft_process_outputs =
      llama_dart_type_mask_has(type_mask, COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
  return llama_dart_exit_track_created(
      speculative.release(), llama_dart_speculative_free_object,
      LLAMA_DART_EXIT_STAGE_SESSION, target_context);
}

LLAMADART_API struct llama_dart_speculative *llama_dart_speculative_init(
    struct llama_model *target_model, struct llama_model *draft_model,
    struct llama_context *target_context,
    struct llama_context_params context_params,
    const struct llama_dart_speculative_params *dart_params) {
  return llama_dart_barrier<llama_dart_speculative *>(nullptr, [=] {
    return llama_dart_speculative_init_impl(target_model, draft_model,
                                            target_context, context_params,
                                            dart_params);
  });
}

LLAMADART_API void
llama_dart_speculative_free(struct llama_dart_speculative *speculative) {
  llama_dart_free_barrier([speculative] {
    llama_dart_exit_release(speculative, llama_dart_speculative_free_object);
  });
}

static void llama_dart_speculative_free_object(void *object) {
  auto *speculative = static_cast<llama_dart_speculative *>(object);
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

static bool llama_dart_speculative_begin_impl(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    const llama_token *prompt, int32_t prompt_count) {
  llama_dart_exit_call call;
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

LLAMADART_API bool llama_dart_speculative_begin(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    const llama_token *prompt, int32_t prompt_count) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_speculative_begin_impl(speculative, seq_id, prompt,
                                             prompt_count);
  });
}

static bool llama_dart_speculative_process_batch_impl(
    struct llama_dart_speculative *speculative, struct llama_batch batch) {
  llama_dart_exit_call call;
  if (speculative == nullptr || speculative->spec == nullptr) {
    return false;
  }
  if (speculative->caps_draft_process_outputs) {
    batch = llama_dart_cap_batch_outputs(batch, speculative->process_output_mask);
  }
  return llama_dart_speculative_process_compat(speculative->spec, batch,
                                              speculative->ctx_tgt,
                                              speculative->ctx_dft);
}

LLAMADART_API bool llama_dart_speculative_process_batch(
    struct llama_dart_speculative *speculative, struct llama_batch batch) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_speculative_process_batch_impl(speculative, batch);
  });
}

static int32_t llama_dart_speculative_draft_impl(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last, const llama_token *prompt,
    int32_t prompt_count, int32_t draft_token_max, llama_token *out_tokens,
    int32_t out_capacity) {
  llama_dart_exit_call call;
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

LLAMADART_API int32_t llama_dart_speculative_draft(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    llama_pos n_past, llama_token id_last, const llama_token *prompt,
    int32_t prompt_count, int32_t draft_token_max, llama_token *out_tokens,
    int32_t out_capacity) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    return llama_dart_speculative_draft_impl(
        speculative, seq_id, n_past, id_last, prompt, prompt_count,
        draft_token_max, out_tokens, out_capacity);
  });
}

static void llama_dart_speculative_accept_impl(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    uint16_t accepted_count) {
  llama_dart_exit_call call;
  if (speculative == nullptr || speculative->spec == nullptr || seq_id != 0) {
    return;
  }
  if (!speculative->has_last_draft) {
    return;
  }
  common_speculative_accept(speculative->spec, seq_id, accepted_count);
  speculative->has_last_draft = false;
}

LLAMADART_API void llama_dart_speculative_accept(
    struct llama_dart_speculative *speculative, llama_seq_id seq_id,
    uint16_t accepted_count) {
  llama_dart_void_barrier([=] {
    llama_dart_speculative_accept_impl(speculative, seq_id, accepted_count);
  });
}

static struct llama_dart_mtp *llama_dart_mtp_init_impl(
    struct llama_model *draft_model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  llama_dart_exit_creating_call call;
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

  // Owns the draft context until the handle does, so that neither a failure
  // nor an exception leaves it behind.
  std::unique_ptr<llama_context, void (*)(llama_context *)> draft_context(
      llama_init_from_model(draft_model, ctx_params), llama_free);
  llama_context *ctx_dft = draft_context.get();
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

  std::unique_ptr<common_speculative, void (*)(common_speculative *)> spec(
      common_speculative_init(params, 1), common_speculative_free);
  if (spec == nullptr) {
    LOG_WRN("%s: failed to initialize common_speculative draft-mtp state\n",
            __func__);
    return nullptr;
  }

  auto *mtp = new llama_dart_mtp();
  mtp->ctx_tgt = ctx_tgt;
  mtp->ctx_dft = draft_context.release();
  mtp->spec = spec.release();
  return llama_dart_exit_track_created(mtp, llama_dart_mtp_free_object,
                                       LLAMA_DART_EXIT_STAGE_SESSION, ctx_tgt);
}

LLAMADART_API struct llama_dart_mtp *llama_dart_mtp_init(
    struct llama_model *model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  return llama_dart_barrier<llama_dart_mtp *>(nullptr, [=] {
    return llama_dart_mtp_init_impl(model, ctx_tgt, ctx_params,
                                    draft_token_max, draft_token_min,
                                    min_probability, backend_sampling);
  });
}

LLAMADART_API struct llama_dart_mtp *llama_dart_mtp_init_with_draft_model(
    struct llama_model *draft_model, struct llama_context *ctx_tgt,
    struct llama_context_params ctx_params, int32_t draft_token_max,
    int32_t draft_token_min, float min_probability, bool backend_sampling) {
  return llama_dart_barrier<llama_dart_mtp *>(nullptr, [=] {
    return llama_dart_mtp_init_impl(draft_model, ctx_tgt, ctx_params,
                                    draft_token_max, draft_token_min,
                                    min_probability, backend_sampling);
  });
}

LLAMADART_API void llama_dart_mtp_free(struct llama_dart_mtp *mtp) {
  llama_dart_free_barrier(
      [mtp] { llama_dart_exit_release(mtp, llama_dart_mtp_free_object); });
}

static void llama_dart_mtp_free_object(void *object) {
  auto *mtp = static_cast<llama_dart_mtp *>(object);
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

static bool llama_dart_mtp_begin_impl(struct llama_dart_mtp *mtp,
                                      llama_seq_id seq_id,
                                      const llama_token *prompt,
                                      int32_t prompt_count) {
  llama_dart_exit_call call;
  if (mtp == nullptr || mtp->spec == nullptr || prompt_count < 0 ||
      !llama_dart_mtp_valid_seq_id(seq_id)) {
    return false;
  }

  mtp->prompt.clear();
  mtp->has_last_draft = false;
  if (prompt != nullptr && prompt_count > 0) {
    mtp->prompt.assign(prompt, prompt + prompt_count);
  }

  common_speculative_begin(mtp->spec, seq_id, mtp->prompt);
  return true;
}

LLAMADART_API bool llama_dart_mtp_begin(struct llama_dart_mtp *mtp,
                                        llama_seq_id seq_id,
                                        const llama_token *prompt,
                                        int32_t prompt_count) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_mtp_begin_impl(mtp, seq_id, prompt, prompt_count);
  });
}

static bool
llama_dart_mtp_process_batch_impl(struct llama_dart_mtp *mtp,
                                  struct llama_batch batch) {
  llama_dart_exit_call call;
  if (mtp == nullptr || mtp->spec == nullptr) {
    return false;
  }
  return llama_dart_speculative_process_compat(mtp->spec, batch, mtp->ctx_tgt,
                                              mtp->ctx_dft);
}

LLAMADART_API bool
llama_dart_mtp_process_batch(struct llama_dart_mtp *mtp,
                             struct llama_batch batch) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_mtp_process_batch_impl(mtp, batch);
  });
}

static int32_t llama_dart_mtp_draft_impl(
    struct llama_dart_mtp *mtp, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  llama_dart_exit_call call;
  if (mtp == nullptr || mtp->spec == nullptr || out_tokens == nullptr ||
      out_capacity < 0 || prompt_count < 0 || draft_token_max <= 0 ||
      !llama_dart_mtp_valid_seq_id(seq_id)) {
    if (mtp != nullptr) {
      mtp->has_last_draft = false;
    }
    return -1;
  }

  mtp->prompt.clear();
  if (prompt != nullptr && prompt_count > 0) {
    mtp->prompt.assign(prompt, prompt + prompt_count);
  }
  mtp->draft.clear();
  mtp->has_last_draft = false;
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

  const int32_t count = llama_dart_mtp_draft_count(
      mtp->draft.size(), draft_token_max, out_capacity);
  mtp->has_last_draft = count > 0;
  for (int32_t i = 0; i < count; ++i) {
    out_tokens[i] = mtp->draft[static_cast<size_t>(i)];
  }
  return count;
}

LLAMADART_API int32_t llama_dart_mtp_draft(
    struct llama_dart_mtp *mtp, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    return llama_dart_mtp_draft_impl(mtp, seq_id, n_past, id_last, prompt,
                                     prompt_count, draft_token_max, out_tokens,
                                     out_capacity);
  });
}

static void llama_dart_mtp_accept_impl(struct llama_dart_mtp *mtp,
                                       llama_seq_id seq_id,
                                       uint16_t accepted_count) {
  llama_dart_exit_call call;
  if (mtp == nullptr || mtp->spec == nullptr) {
    return;
  }
  if (!llama_dart_mtp_valid_seq_id(seq_id) ||
      !llama_dart_mtp_take_last_draft(mtp)) {
    return;
  }
  common_speculative_accept(mtp->spec, seq_id, accepted_count);
}

LLAMADART_API void llama_dart_mtp_accept(struct llama_dart_mtp *mtp,
                                         llama_seq_id seq_id,
                                         uint16_t accepted_count) {
  llama_dart_void_barrier(
      [=] { llama_dart_mtp_accept_impl(mtp, seq_id, accepted_count); });
}

static struct llama_dart_ngram *
llama_dart_ngram_simple_init_impl(int32_t ngram_size, int32_t draft_token_max) {
  llama_dart_exit_creating_call call;
  common_params_speculative params;
  params.types = {COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE};
  params.ngram_simple.size_n = llama_dart_uint16_or_default(ngram_size, 12);
  params.ngram_simple.size_m =
      llama_dart_uint16_or_default(draft_token_max, 48);

  std::unique_ptr<common_speculative, void (*)(common_speculative *)> spec(
      common_speculative_init(params, 1), common_speculative_free);
  if (spec == nullptr) {
    LOG_WRN("%s: failed to initialize common_speculative ngram-simple state\n",
            __func__);
    return nullptr;
  }

  auto *ngram = new llama_dart_ngram();
  ngram->spec = spec.release();
  return llama_dart_exit_track_created(ngram, llama_dart_ngram_free_object,
                                       LLAMA_DART_EXIT_STAGE_SESSION);
}

LLAMADART_API struct llama_dart_ngram *
llama_dart_ngram_simple_init(int32_t ngram_size, int32_t draft_token_max) {
  return llama_dart_barrier<struct llama_dart_ngram *>(nullptr, [=] {
    return llama_dart_ngram_simple_init_impl(ngram_size, draft_token_max);
  });
}

LLAMADART_API void llama_dart_ngram_free(struct llama_dart_ngram *ngram) {
  llama_dart_free_barrier([ngram] {
    llama_dart_exit_release(ngram, llama_dart_ngram_free_object);
  });
}

static void llama_dart_ngram_free_object(void *object) {
  auto *ngram = static_cast<llama_dart_ngram *>(object);
  if (ngram->spec != nullptr) {
    common_speculative_free(ngram->spec);
    ngram->spec = nullptr;
  }
  delete ngram;
}

static bool llama_dart_ngram_begin_impl(struct llama_dart_ngram *ngram,
                                        llama_seq_id seq_id,
                                        const llama_token *prompt,
                                        int32_t prompt_count) {
  llama_dart_exit_call call;
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

LLAMADART_API bool llama_dart_ngram_begin(struct llama_dart_ngram *ngram,
                                          llama_seq_id seq_id,
                                          const llama_token *prompt,
                                          int32_t prompt_count) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_ngram_begin_impl(ngram, seq_id, prompt, prompt_count);
  });
}

static bool
llama_dart_ngram_process_batch_impl(struct llama_dart_ngram *ngram,
                                    struct llama_batch batch) {
  llama_dart_exit_call call;
  if (ngram == nullptr || ngram->spec == nullptr) {
    return false;
  }
  return llama_dart_speculative_process_compat(ngram->spec, batch, nullptr);
}

LLAMADART_API bool
llama_dart_ngram_process_batch(struct llama_dart_ngram *ngram,
                               struct llama_batch batch) {
  return llama_dart_barrier<bool>(false, [=] {
    return llama_dart_ngram_process_batch_impl(ngram, batch);
  });
}

static int32_t llama_dart_ngram_draft_impl(
    struct llama_dart_ngram *ngram, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  llama_dart_exit_call call;
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

LLAMADART_API int32_t llama_dart_ngram_draft(
    struct llama_dart_ngram *ngram, llama_seq_id seq_id, llama_pos n_past,
    llama_token id_last, const llama_token *prompt, int32_t prompt_count,
    int32_t draft_token_max, llama_token *out_tokens, int32_t out_capacity) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    return llama_dart_ngram_draft_impl(ngram, seq_id, n_past, id_last, prompt,
                                       prompt_count, draft_token_max,
                                       out_tokens, out_capacity);
  });
}

static void llama_dart_ngram_accept_impl(struct llama_dart_ngram *ngram,
                                         llama_seq_id seq_id,
                                         uint16_t accepted_count) {
  llama_dart_exit_call call;
  if (ngram == nullptr || ngram->spec == nullptr) {
    return;
  }
  if (seq_id != 0 || !ngram->has_last_draft) {
    return;
  }
  common_speculative_accept(ngram->spec, seq_id, accepted_count);
  ngram->has_last_draft = false;
}

LLAMADART_API void llama_dart_ngram_accept(struct llama_dart_ngram *ngram,
                                           llama_seq_id seq_id,
                                           uint16_t accepted_count) {
  llama_dart_void_barrier(
      [=] { llama_dart_ngram_accept_impl(ngram, seq_id, accepted_count); });
}

static int32_t llama_dart_sampler_sample_and_accept_n_impl(
    struct llama_sampler *sampler, struct llama_context *ctx,
    const int32_t *idxs, int32_t idx_count, const llama_token *draft_tokens,
    int32_t draft_count, llama_token *out_tokens, int32_t out_capacity) {
  llama_dart_exit_call call;
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

LLAMADART_API int32_t llama_dart_sampler_sample_and_accept_n(
    struct llama_sampler *sampler, struct llama_context *ctx,
    const int32_t *idxs, int32_t idx_count, const llama_token *draft_tokens,
    int32_t draft_count, llama_token *out_tokens, int32_t out_capacity) {
  return llama_dart_barrier<int32_t>(LLAMA_DART_STATUS_EXCEPTION, [=] {
    return llama_dart_sampler_sample_and_accept_n_impl(
        sampler, ctx, idxs, idx_count, draft_tokens, draft_count, out_tokens,
        out_capacity);
  });
}
}
