#pragma once

#include "common.h"

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
