#pragma once

#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cstdint>
#include <vector>

struct llama_dart_mtp {
  llama_context *ctx_tgt = nullptr;
  llama_context *ctx_dft = nullptr;
  common_speculative *spec = nullptr;
  std::vector<llama_token> prompt;
  std::vector<llama_token> draft;
  bool has_last_draft = false;
};

static inline bool llama_dart_mtp_valid_seq_id(llama_seq_id seq_id) {
  return seq_id == 0;
}

static inline int32_t llama_dart_mtp_draft_count(size_t draft_size,
                                                 int32_t draft_token_max,
                                                 int32_t out_capacity) {
  if (draft_token_max <= 0 || out_capacity <= 0) {
    return 0;
  }
  return static_cast<int32_t>(
      std::min(draft_size,
               static_cast<size_t>(std::min(draft_token_max, out_capacity))));
}

static inline bool llama_dart_mtp_take_last_draft(llama_dart_mtp *mtp) {
  if (mtp == nullptr || !mtp->has_last_draft) {
    return false;
  }
  mtp->has_last_draft = false;
  return true;
}
