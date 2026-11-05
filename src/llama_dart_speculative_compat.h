#pragma once

#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// Rollback probing clears memory and decodes dummy tokens. Check
// every sequence before allowing that probe to run on a caller-owned target.
inline bool llama_dart_mtp_target_is_empty(llama_context *context) {
  const auto memory = llama_get_memory(context);
  if (memory == nullptr) {
    return false;
  }
  for (uint32_t sequence = 0; sequence < llama_n_seq_max(context); ++sequence) {
    if (llama_memory_seq_pos_max(memory, sequence) >= 0) {
      return false;
    }
  }
  return true;
}

// A shared-KV MTP draft reads the target's cache, so both probe decodes must
// finish before either cache is rolled back or cleared.
inline std::pair<common_context_seq_rm_type, common_context_seq_rm_type>
llama_dart_mtp_probe_rollback(llama_context *target, llama_context *draft) {
#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
  const auto target_memory = llama_get_memory(target);
  const auto draft_memory = llama_get_memory(draft);
  const auto no = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
  if (!target_memory || !draft_memory) {
    return {no, no};
  }
  struct cleanup {
    llama_context *target;
    llama_context *draft;
    ~cleanup() {
      llama_synchronize(target);
      llama_synchronize(draft);
      llama_memory_clear(llama_get_memory(draft), true);
      llama_memory_clear(llama_get_memory(target), true);
    }
  } clear_on_exit{target, draft};

  const auto decode = [](llama_context *context) {
    const llama_tokens tokens = {0, 0};
    common_batch batch = common_batch_get_one(context, tokens);
#if LLAMADART_SPECULATIVE_HAS_BATCH_STATE
    std::vector<float> zeros(llama_model_n_embd_out(llama_get_model(context)),
                             0.0f);
    for (int32_t index = 0; index < batch.size(); ++index) {
      batch.set_embd_state(index, {zeros.data(), 1, zeros.size()});
    }
#endif
    return llama_process(context, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
  };
  if (!decode(target)) {
    return {no, no};
  }
  if (!decode(draft)) {
    return {no, no};
  }
  const auto rollback = [](llama_context *context) {
    if (!llama_memory_seq_rm(llama_get_memory(context), 0, 1, -1)) {
      return COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    }
    return llama_n_rs_seq(context) > 0 ? COMMON_CONTEXT_SEQ_RM_TYPE_RS
                                     : COMMON_CONTEXT_SEQ_RM_TYPE_PART;
  };
  // A shared draft's seq_rm is a no-op; check it while both positions still
  // exist in the target, then check the target's real rollback.
  const auto draft_type = rollback(draft);
  const auto target_type = rollback(target);
  return {target_type, draft_type};
#else
  return {common_context_can_seq_rm(target), common_context_can_seq_rm(draft)};
#endif
}

// Upstream restricts the expert cache to the target context. Keep draft
// microbatch sizing independent: it also controls DFlash's attention mask.
inline void llama_dart_disable_draft_moe_cache(llama_context_params &params) {
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
  params.moe_cache_size = 0;
#else
  (void)params;
#endif
}

#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
// Speculative processing consumes the entries, never renders this batch.
// Preserve llama_batch's implicit sequence/output defaults. A draft context
// supplies the next text position when the caller omits its position array.
inline bool llama_dart_copy_process_batch(llama_batch source, int32_t n_embd,
                                          int32_t n_pos, common_batch &result,
                                          llama_pos first_position = 0) {
  result.clear();
  if (source.n_tokens < 0 || (n_pos != 1 && n_pos != GGML_MROPE_SECTIONS)) {
    return false;
  }
  if (source.n_tokens == 0) {
    return true;
  }
  if ((!source.token && !source.embd) || (source.seq_id && !source.n_seq_id) ||
      (source.embd && (n_embd <= 0 || (n_pos > 1 && !source.pos)))) {
    return false;
  }
  for (int32_t i = 0; i < source.n_tokens; ++i) {
    if (!source.seq_id) {
      continue;
    }
    if (source.n_seq_id[i] <= 0 || !source.seq_id[i]) {
      return false;
    }
    // All public wrapper speculative handles currently own one sequence.
    for (int32_t j = 0; j < source.n_seq_id[i]; ++j) {
      if (source.seq_id[i][j] != 0) {
        return false;
      }
    }
  }
  result.n_pos = source.embd ? n_pos : 1;
  result.tokens.reserve(source.n_tokens);
  for (int32_t i = 0; i < source.n_tokens; ++i) {
    common_batch::token entry{};
    entry.id = source.token ? source.token[i] : LLAMA_TOKEN_NULL;
    for (int32_t p = 0; p < result.n_pos; ++p) {
      entry.pos[p] =
          source.pos ? source.pos[static_cast<size_t>(p) * source.n_tokens + i]
                     : first_position + i;
    }
    entry.seq_id = source.seq_id ? source.seq_id[i][0] : 0;
    if (source.seq_id) {
      entry.seq_ids_extra.assign(source.seq_id[i] + 1,
                                 source.seq_id[i] + source.n_seq_id[i]);
    }
    entry.output = source.logits ? source.logits[i] != 0
                                 : source.embd || i == source.n_tokens - 1;
    if (source.embd) {
      entry.embd = {source.embd + static_cast<size_t>(i) * n_embd, 1,
                    static_cast<size_t>(n_embd)};
    }
    result.tokens.push_back(std::move(entry));
  }
  return true;
}
#endif

inline bool llama_dart_speculative_process_compat(
    common_speculative *spec, llama_batch batch, llama_context *target_context,
    llama_context *draft_context = nullptr) {
#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
  int32_t n_embd = 0;
  int32_t n_pos = 1;
  if (batch.embd && target_context) {
    const auto *model = llama_get_model(target_context);
    n_embd = llama_model_n_embd_inp(model);
    const auto rope = llama_model_rope_type(model);
    if (rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE) {
      n_pos = GGML_MROPE_SECTIONS;
    }
  }
  common_batch converted;
  llama_pos first_position = 0;
  if (!batch.pos && draft_context) {
    const auto memory = llama_get_memory(draft_context);
    if (memory) {
      first_position = llama_memory_seq_pos_max(memory, 0) + 1;
    }
  }
  if (!llama_dart_copy_process_batch(batch, n_embd, n_pos, converted,
                                     first_position)) {
    return false;
  }
  return common_speculative_process(spec, converted);
#else
  (void)target_context;
  (void)draft_context;
  return common_speculative_process(spec, batch);
#endif
}

struct llama_dart_speculative_embedding_requirements {
  bool need_embd_nextn = false;
};

// Preserve the historical next-token embedding query after upstream removed
// common_speculative_need_embd_nextn. Upstream speculative implementations
// now enable their required target outputs during initialization.
inline llama_dart_speculative_embedding_requirements
llama_dart_speculative_embedding_requirements_for(
    const std::vector<common_speculative_type> &types) {
  llama_dart_speculative_embedding_requirements result;
  for (const auto type : types) {
    if (type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
      result.need_embd_nextn = true;
    }
  }
  return result;
}

inline void llama_dart_apply_speculative_draft_output_limits(
    llama_context_params &context_params,
    const std::vector<common_speculative_type> &types, int32_t draft_token_max,
    bool backend_sampling) {
  context_params.n_outputs_max = 1;
  context_params.n_outputs_max_per_seq = 1;

  const bool has_block_draft =
      std::any_of(types.begin(), types.end(), [](common_speculative_type type) {
        return type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
               type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
      });
  if (!has_block_draft) {
    return;
  }

  const uint32_t block_outputs =
      1u + static_cast<uint32_t>(std::max(0, draft_token_max));
  context_params.n_outputs_max = block_outputs;
  if (backend_sampling) {
    context_params.n_outputs_max_per_seq = block_outputs;
  }
}
