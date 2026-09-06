#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_mtp_internal.h"
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

static llama_dart_mtp *make_mtp_contract_backend(common_speculative_type type) {
  common_params_speculative params;
  params.types = {type};
  params.ngram_simple.size_n = 1;
  params.ngram_simple.size_m = 4;
  params.ngram_map_k.size_n = 1;
  params.ngram_map_k.size_m = 4;
  params.ngram_map_k.min_hits = 1;

  auto *mtp = new llama_dart_mtp();
  mtp->spec = common_speculative_init(params, 1);
  assert(mtp->spec != nullptr);
  return mtp;
}

static const std::vector<llama_token> kDraftPrompt = {
    99, 10, 11, 12, 13, 14, 42, 43, 44, 45, 46, 47,
};

static int32_t mtp_contract_draft(llama_dart_mtp *mtp, int32_t draft_token_max,
                                  llama_token *out_tokens,
                                  int32_t out_capacity) {
  return llama_dart_mtp_draft(
      mtp, 0, static_cast<llama_pos>(kDraftPrompt.size()), 42,
      kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()),
      draft_token_max, out_tokens, out_capacity);
}

static void test_mtp_accept_requires_draft() {
  auto *mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  llama_dart_mtp_accept(mtp, 0, 1);
  llama_dart_mtp_free(mtp);

  mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
  const std::vector<llama_token> no_match = {1, 2, 3, 4, 5, 6, 7, 8};
  llama_token output[4] = {};
  assert(llama_dart_mtp_draft(mtp, 0, static_cast<llama_pos>(no_match.size()),
                              99, no_match.data(),
                              static_cast<int32_t>(no_match.size()), 4, output,
                              4) == 0);
  llama_dart_mtp_accept(mtp, 0, 1);
  llama_dart_mtp_free(mtp);
}

static void test_mtp_draft_clamps_and_failed_draft() {
  assert(llama_dart_mtp_draft_count(8, 2, 6) == 2);
  assert(llama_dart_mtp_draft_count(8, 6, 3) == 3);

  auto *mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_mtp_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  llama_token output[6] = {};
  assert(mtp_contract_draft(mtp, 6, output, 2) == 2);
  llama_dart_mtp_free(mtp);

  mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_mtp_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  assert(mtp_contract_draft(mtp, 2, output, 6) == 2);
  llama_dart_mtp_free(mtp);

  mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_mtp_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  assert(mtp_contract_draft(mtp, 4, output, 6) == 4);
  assert(mtp->has_last_draft);
  assert(mtp_contract_draft(mtp, 4, nullptr, 6) == -1);
  assert(!mtp->has_last_draft);
  llama_dart_mtp_accept(mtp, 0, 0);
  assert(!mtp->has_last_draft);
  llama_dart_mtp_free(mtp);
}

static void test_mtp_repeated_accept_and_sequence_validation() {
  llama_dart_mtp lifecycle;
  lifecycle.has_last_draft = true;
  assert(llama_dart_mtp_take_last_draft(&lifecycle));
  assert(!llama_dart_mtp_take_last_draft(&lifecycle));

  auto *mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(!llama_dart_mtp_begin(mtp, -1, kDraftPrompt.data(),
                               static_cast<int32_t>(kDraftPrompt.size())));
  assert(!llama_dart_mtp_begin(mtp, 1, kDraftPrompt.data(),
                               static_cast<int32_t>(kDraftPrompt.size())));
  assert(llama_dart_mtp_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));

  llama_token output[4] = {};
  assert(llama_dart_mtp_draft(
             mtp, -1, static_cast<llama_pos>(kDraftPrompt.size()), 42,
             kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()), 4,
             output, 4) == -1);
  assert(llama_dart_mtp_draft(
             mtp, 1, static_cast<llama_pos>(kDraftPrompt.size()), 42,
             kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()), 4,
             output, 4) == -1);

  assert(mtp_contract_draft(mtp, 4, output, 4) == 4);
  assert(mtp->has_last_draft);
  llama_dart_mtp_accept(mtp, -1, 0);
  llama_dart_mtp_accept(mtp, 1, 0);
  assert(mtp->has_last_draft);
  llama_dart_mtp_accept(mtp, 0, 1);
  assert(!mtp->has_last_draft);
  llama_dart_mtp_accept(mtp, 0, 0);
  assert(!mtp->has_last_draft);
  llama_dart_mtp_free(mtp);
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
  test_mtp_accept_requires_draft();
  test_mtp_draft_clamps_and_failed_draft();
  test_mtp_repeated_accept_and_sequence_validation();
  return 0;
}
