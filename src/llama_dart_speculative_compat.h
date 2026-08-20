#pragma once

#include "common.h"

#include <algorithm>
#include <cstdint>
#include <vector>

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
