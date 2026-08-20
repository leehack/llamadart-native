#include "llama_dart_speculative_compat.h"
#include "llama_dart_wrapper.h"

#include <cassert>
#include <vector>

static void expect_requirements(common_speculative_type type, bool need_embd,
                                bool need_embd_nextn) {
  const auto requirements =
      llama_dart_speculative_embedding_requirements_for({type});
  assert(requirements.need_embd == need_embd);
  assert(requirements.need_embd_nextn == need_embd_nextn);
}

static void expect_draft_output_limits(
    std::vector<common_speculative_type> types, int32_t draft_token_max,
    bool backend_sampling, uint32_t expected_total,
    uint32_t expected_per_sequence) {
  auto params = llama_context_default_params();
  params.n_outputs_max = 99;
  params.n_outputs_max_per_seq = 99;

  llama_dart_apply_speculative_draft_output_limits(
      params, types, draft_token_max, backend_sampling);

  assert(params.n_outputs_max == expected_total);
  assert(params.n_outputs_max_per_seq == expected_per_sequence);
}

int main() {
  assert(!llama_dart_speculative_need_embd(nullptr));
  assert(!llama_dart_speculative_need_embd_nextn(nullptr));

  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, false, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, false, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, false, true);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, false, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, false, false);

  const auto combined = llama_dart_speculative_embedding_requirements_for({
      COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
      COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
  });
  assert(!combined.need_embd);
  assert(combined.need_embd_nextn);

  expect_draft_output_limits({COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE}, 7, true, 1,
                             1);
  expect_draft_output_limits({COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH}, 7, false,
                             8, 1);
  expect_draft_output_limits({COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK}, 7, true, 8,
                             8);
  expect_draft_output_limits(
      {COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
      3, true, 4, 4);
  return 0;
}
