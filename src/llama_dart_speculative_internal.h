#pragma once

#include "llama_dart_session_lifetime_internal.h"
#include "llama_dart_speculative_compat.h"
#include "llama_dart_wrapper.h"
#include "speculative.h"

#include <cstdint>
#include <vector>

struct llama_dart_speculative {
  llama_dart_internal::session_lifetime lifetime{
      this, llama_dart_internal::session_kind::speculative};
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
