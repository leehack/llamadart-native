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

#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
static void test_legacy_process_batch() {
  auto batch = llama_batch_init(2, 0, 1);
  batch.n_tokens = 2;
  for (int32_t i = 0; i < 2; ++i) {
    batch.token[i] = 42 + i;
    batch.pos[i] = 8 + i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = i == 0;
  }
  common_batch converted;
  assert(llama_dart_copy_process_batch(batch, 0, 1, converted));
  assert(converted.size() == 2 && converted.n_pos == 1);
  for (int32_t i = 0; i < 2; ++i) {
    const auto &entry = converted.tokens[i];
    assert(entry.id == batch.token[i] && entry.pos[0] == batch.pos[i]);
    assert(entry.seq_id == 0 && entry.seq_ids_extra.empty());
    assert(entry.output == (i == 0) && entry.embd.data == nullptr);
  }
  auto missing = batch;
  missing.logits = nullptr;
  assert(llama_dart_copy_process_batch(missing, 0, 1, converted));
  assert(!converted.tokens[0].output && converted.tokens[1].output);
  missing = batch;
  missing.pos = nullptr;
  missing.seq_id = nullptr;
  missing.n_seq_id = nullptr;
  assert(llama_dart_copy_process_batch(missing, 0, 1, converted, 15));
  assert(converted.tokens[0].pos[0] == 15 && converted.tokens[1].pos[0] == 16);
  assert(converted.tokens[0].seq_id == 0 && converted.tokens[1].seq_id == 0);
  missing = batch;
  missing.token = nullptr;
  assert(!llama_dart_copy_process_batch(missing, 0, 1, converted));
  missing = batch;
  missing.n_seq_id = nullptr;
  assert(!llama_dart_copy_process_batch(missing, 0, 1, converted));
  batch.n_seq_id[0] = 0;
  assert(!llama_dart_copy_process_batch(batch, 0, 1, converted));
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 1;
  assert(!llama_dart_copy_process_batch(batch, 0, 1, converted));
  batch.seq_id[0][0] = 0;

  // Exercise the actual exported MTP process path with a model-free n-gram
  // backend; malformed input must be rejected before reaching upstream.
  auto *mtp = make_mtp_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
  assert(llama_dart_mtp_process_batch(mtp, batch));
  assert(!llama_dart_mtp_process_batch(mtp, missing));
  llama_dart_mtp_free(mtp);
  auto *ngram = llama_dart_ngram_simple_init(1, 4);
  assert(ngram != nullptr);
  assert(llama_dart_ngram_process_batch(ngram, batch));
  assert(!llama_dart_ngram_process_batch(ngram, missing));
  llama_dart_ngram_free(ngram);
  llama_batch_free(batch);

  float embeddings[] = {1, 2, 3, 4, 5, 6};
  llama_pos positions[] = {10, 11, 20, 21, 30, 31, 40, 41};
  int32_t counts[] = {1, 1};
  llama_seq_id sequence = 0;
  llama_seq_id *sequences[] = {&sequence, &sequence};
  int8_t outputs[] = {0, 1};
  llama_batch media{2, nullptr, embeddings, positions, counts, sequences, outputs};
  assert(llama_dart_copy_process_batch(media, 3, 4, converted));
  assert(converted.n_pos == 4);
  for (int32_t i = 0; i < 2; ++i) {
    const auto &entry = converted.tokens[i];
    assert(entry.id == LLAMA_TOKEN_NULL);
    assert(entry.embd.data == embeddings + i * 3);
    assert(entry.embd.n_rows == 1 && entry.embd.n_embd == 3);
    for (int32_t p = 0; p < 4; ++p) {
      assert(entry.pos[p] == positions[p * 2 + i]);
    }
    assert(entry.output == (i == 1));
  }
  assert(!llama_dart_copy_process_batch(media, 0, 4, converted));
  media.logits = nullptr;
  assert(llama_dart_copy_process_batch(media, 3, 4, converted));
  assert(converted.tokens[0].output && converted.tokens[1].output);
  auto missing_media_position = media;
  missing_media_position.pos = nullptr;
  assert(!llama_dart_copy_process_batch(missing_media_position, 3, 4, converted));
  assert(!llama_dart_copy_process_batch(media, 3, 2, converted));
  media.n_tokens = -1;
  assert(!llama_dart_copy_process_batch(media, 3, 4, converted));
  assert(llama_dart_copy_process_batch(llama_batch{}, 0, 1, converted));
  assert(converted.size() == 0);
}
#endif

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
#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
  test_legacy_process_batch();
#endif
  return 0;
}
