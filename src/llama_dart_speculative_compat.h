#pragma once

#include "common.h"
#include "speculative.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

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
  bool need_embd = false;
  bool need_embd_nextn = false;
};

// Preserve the historical libllamadart answers after upstream removed its
// common_speculative_need_embd* queries. Upstream speculative implementations
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
