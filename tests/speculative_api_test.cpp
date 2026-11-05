#ifdef NDEBUG
#error "Wrapper contract tests require active assertions in every configuration"
#endif

#include "llama_dart_speculative_internal.h"
#include "llama_dart_speculative_compat.h"
#include "llama_dart_wrapper.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

template <typename T, typename = void>
struct has_moe_cache_field : std::false_type {};

template <typename T>
struct has_moe_cache_field<T, std::void_t<decltype(T::moe_cache_size)>>
    : std::true_type {};

// A failed/miswired build probe must not silently compile the guard as a no-op
// when these headers actually contain the cache budget.
static_assert(bool(LLAMADART_CONTEXT_HAS_MOE_CACHE) ==
              has_moe_cache_field<llama_context_params>::value);

#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
template <typename T, typename = void>
struct has_batch_state : std::false_type {};

template <typename T>
struct has_batch_state<T, std::void_t<decltype(&T::set_embd_state)>>
    : std::true_type {};

static_assert(bool(LLAMADART_SPECULATIVE_HAS_BATCH_STATE) ==
              has_batch_state<common_batch>::value);
#endif

static void test_draft_moe_cache() {
  auto target = llama_context_default_params();
  target.n_batch = 512;
  target.n_ubatch = 128;
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
  target.moe_cache_size = 64 * 1024 * 1024;
#endif
  auto draft = target;
  llama_dart_disable_draft_moe_cache(draft);
#if LLAMADART_CONTEXT_HAS_MOE_CACHE
  assert(draft.moe_cache_size == 0);
  assert(target.moe_cache_size == 64 * 1024 * 1024);
#endif
  assert(draft.n_batch == 512 && draft.n_ubatch == 128);
  // The same operation also accepts the already-disabled/default case.
  llama_dart_disable_draft_moe_cache(draft);
  assert(draft.n_batch == 512 && draft.n_ubatch == 128);
}

static void expect_requirements(common_speculative_type type,
                                bool need_embd_nextn) {
  const auto requirements =
      llama_dart_speculative_embedding_requirements_for({type});
  assert(requirements.need_embd_nextn == need_embd_nextn);
}

static void test_nextn_query_export() {
  for (const auto types : {
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
           std::vector<common_speculative_type>{COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
                                                COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
       }) {
    auto *session = new llama_dart_speculative();
    session->embedding_requirements =
        llama_dart_speculative_embedding_requirements_for(types);
    assert(llama_dart_speculative_need_embd_nextn(session) ==
           session->embedding_requirements.need_embd_nextn);
    llama_dart_speculative_free(session);
  }
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

static llama_dart_speculative *make_generic_contract_backend(common_speculative_type type) {
  common_params_speculative params;
  params.types = {type};
  params.ngram_simple.size_n = 1;
  params.ngram_simple.size_m = 4;
  params.ngram_map_k.size_n = 1;
  params.ngram_map_k.size_m = 4;
  params.ngram_map_k.min_hits = 1;

  auto *mtp = new llama_dart_speculative();
  mtp->spec = common_speculative_init(params, 1);
  assert(mtp->spec != nullptr);
  return mtp;
}

static const std::vector<llama_token> kDraftPrompt = {
    99, 10, 11, 12, 13, 14, 42, 43, 44, 45, 46, 47,
};

#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
static void test_generic_process_batch() {
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

  // Both draftless backends use the exported generic batch conversion path.
  for (auto type : {COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
                    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE}) {
    auto *session = make_generic_contract_backend(type);
    assert(llama_dart_speculative_process_batch(session, batch));
    assert(!llama_dart_speculative_process_batch(session, missing));
    llama_dart_speculative_free(session);
  }
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

static int32_t generic_contract_draft(llama_dart_speculative *mtp, int32_t draft_token_max,
                                  llama_token *out_tokens,
                                  int32_t out_capacity) {
  return llama_dart_speculative_draft(
      mtp, 0, static_cast<llama_pos>(kDraftPrompt.size()), 42,
      kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()),
      draft_token_max, out_tokens, out_capacity);
}

static void test_generic_accept_requires_draft() {
  auto *mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  llama_dart_speculative_accept(mtp, 0, 1);
  llama_dart_speculative_free(mtp);

  mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
  const std::vector<llama_token> no_match = {1, 2, 3, 4, 5, 6, 7, 8};
  llama_token output[4] = {};
  assert(llama_dart_speculative_draft(mtp, 0, static_cast<llama_pos>(no_match.size()),
                              99, no_match.data(),
                              static_cast<int32_t>(no_match.size()), 4, output,
                              4) == 0);
  llama_dart_speculative_accept(mtp, 0, 1);
  llama_dart_speculative_free(mtp);
}

static void test_generic_draft_clamps_and_failed_draft() {
  auto *mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_speculative_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  llama_token output[6] = {};
  assert(generic_contract_draft(mtp, 6, output, 2) == 2);
  llama_dart_speculative_free(mtp);

  mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_speculative_begin(mtp, 0, kDraftPrompt.data(),
                                      static_cast<int32_t>(kDraftPrompt.size())));
  assert(generic_contract_draft(mtp, 6, output, 3) == 3);
  llama_dart_speculative_free(mtp);

  mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_speculative_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  assert(generic_contract_draft(mtp, 2, output, 6) == 2);
  llama_dart_speculative_free(mtp);

  mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(llama_dart_speculative_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));
  assert(generic_contract_draft(mtp, 4, output, 6) == 4);
  assert(mtp->has_last_draft);
  assert(generic_contract_draft(mtp, 4, nullptr, 6) == -1);
  assert(!mtp->has_last_draft);
  llama_dart_speculative_accept(mtp, 0, 0);
  assert(!mtp->has_last_draft);
  llama_dart_speculative_free(mtp);
}

static void test_generic_repeated_accept_and_sequence_validation() {
  auto *mtp = make_generic_contract_backend(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
  assert(!llama_dart_speculative_begin(mtp, -1, kDraftPrompt.data(),
                               static_cast<int32_t>(kDraftPrompt.size())));
  assert(!llama_dart_speculative_begin(mtp, 1, kDraftPrompt.data(),
                               static_cast<int32_t>(kDraftPrompt.size())));
  assert(llama_dart_speculative_begin(mtp, 0, kDraftPrompt.data(),
                              static_cast<int32_t>(kDraftPrompt.size())));

  llama_token output[4] = {};
  assert(llama_dart_speculative_draft(
             mtp, -1, static_cast<llama_pos>(kDraftPrompt.size()), 42,
             kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()), 4,
             output, 4) == -1);
  assert(llama_dart_speculative_draft(
             mtp, 1, static_cast<llama_pos>(kDraftPrompt.size()), 42,
             kDraftPrompt.data(), static_cast<int32_t>(kDraftPrompt.size()), 4,
             output, 4) == -1);

  assert(generic_contract_draft(mtp, 4, output, 4) == 4);
  assert(mtp->has_last_draft);
  llama_dart_speculative_accept(mtp, -1, 0);
  llama_dart_speculative_accept(mtp, 1, 0);
  assert(mtp->has_last_draft);
  llama_dart_speculative_accept(mtp, 0, 1);
  assert(!mtp->has_last_draft);
  llama_dart_speculative_accept(mtp, 0, 0);
  assert(!mtp->has_last_draft);
  llama_dart_speculative_free(mtp);
}

// Optional model-backed guard qualification: a small ordinary decoder is
// enough to check caller-state preservation, but does not qualify MTP output.
static void test_mtp_fresh_target_contract(const char *model_path,
                                          const char *mtp_path) {
  llama_backend_init();
  auto model_params = llama_model_default_params();
  ggml_backend_dev_t devices[] = {nullptr};
  model_params.devices = devices;
  model_params.n_gpu_layers = 0;
  auto *model = llama_model_load_from_file(model_path, model_params);
  assert(model != nullptr);
  llama_model *draft_model = nullptr;
  if (mtp_path != nullptr) {
    model_params.load_mtp = true;
    draft_model = llama_model_load_from_file(mtp_path, model_params);
    assert(draft_model != nullptr);
  }
  auto params = llama_context_default_params();
  params.n_ctx = 256;
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  params.n_batch = 32;
  params.n_ubatch = 32;
  params.n_seq_max = 2;
  params.n_threads = 2;
  params.n_threads_batch = 2;
  auto *target = llama_init_from_model(model, params);
  assert(target != nullptr);
  const auto memory = llama_get_memory(target);
  assert(llama_dart_mtp_target_is_empty(target));

  auto batch = llama_batch_init(2, 0, 1);
  batch.n_tokens = 2;
  for (int32_t i = 0; i < 2; ++i) {
    batch.token[i] = 1;
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.logits[i] = i == 1;
  }
  for (llama_seq_id sequence = 0; sequence < 2; ++sequence) {
    for (int32_t i = 0; i < 2; ++i) {
      batch.seq_id[i][0] = sequence;
    }
    assert(llama_decode(target, batch) == 0);
    assert(!llama_dart_mtp_target_is_empty(target));
    const size_t state_size = llama_state_get_size(target);
    std::vector<uint8_t> before(state_size), after(state_size);
    assert(llama_state_get_data(target, before.data(), before.size()) ==
           state_size);
    llama_dart_speculative_params speculative_params{};
    speculative_params.type_names = "draft-mtp";
    auto *speculative = llama_dart_speculative_init(
        model, draft_model, target, params, &speculative_params);
    assert(speculative == nullptr);
    assert(llama_state_get_size(target) == state_size);
    assert(llama_state_get_data(target, after.data(), after.size()) ==
           state_size);
    assert(before == after);
    assert(llama_memory_seq_pos_max(memory, sequence) == 1);

    // These strategies already support populated targets. The MTP probe must
    // not erase their prompt or impose its initialization precondition.
    for (const auto *type : {"ngram-simple", "draft-simple"}) {
      speculative_params.type_names = type;
      speculative = llama_dart_speculative_init(model, model, target, params,
                                                &speculative_params);
      assert(speculative != nullptr);
      assert(llama_memory_seq_pos_max(memory, sequence) == 1);
      llama_dart_speculative_free(speculative);
      assert(llama_memory_seq_pos_max(memory, sequence) == 1);
    }
    llama_memory_clear(memory, true);
    assert(llama_dart_mtp_target_is_empty(target));
  }
  llama_batch_free(batch);
  llama_free(target);
  if (draft_model != nullptr) {
    llama_model_free(draft_model);
  }
  llama_model_free(model);
  llama_backend_free();
  std::puts("MTP fresh-target rejection and non-MTP preservation passed (not "
            "MTP model qualification)");
}

static void test_mtp_generation(llama_model *model, llama_context *target,
                                llama_context_params params,
                                llama_dart_speculative *speculative) {
  constexpr int32_t output_count = 16;
  const auto greedy = [](llama_context *context, int32_t index) {
    const float *logits = llama_get_logits_ith(context, index);
    assert(logits != nullptr);
    const auto vocabulary = llama_model_get_vocab(llama_get_model(context));
    const int32_t count = llama_vocab_n_tokens(vocabulary);
    return static_cast<llama_token>(
        std::max_element(logits, logits + count) - logits);
  };
  const auto decode = [](llama_context *context,
                         const std::vector<llama_token> &tokens,
                         llama_pos position,
                         llama_dart_speculative *spec) {
    auto batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
      batch.token[i] = tokens[i];
      batch.pos[i] = position + i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = true;
    }
    assert(llama_decode(context, batch) == 0);
    if (spec) {
      assert(llama_dart_speculative_process_batch(spec, batch));
    }
    llama_batch_free(batch);
  };

  std::vector<llama_token> prompt = {1, 3, 4, 5};
  auto *baseline = llama_init_from_model(model, params);
  assert(baseline != nullptr);
  decode(baseline, prompt, 0, nullptr);
  std::vector<llama_token> expected;
  auto next = greedy(baseline, -1);
  for (int32_t i = 0; i < output_count; ++i) {
    expected.push_back(next);
    decode(baseline, {next}, static_cast<llama_pos>(prompt.size()) + i,
           nullptr);
    next = greedy(baseline, -1);
  }
  llama_free(baseline);

  assert(llama_dart_mtp_target_is_empty(target));
  decode(target, prompt, 0, speculative);
  assert(llama_dart_speculative_begin(speculative, 0, prompt.data(),
                                      static_cast<int32_t>(prompt.size())));
  next = greedy(target, -1);
  auto past = static_cast<llama_pos>(prompt.size());
  std::vector<llama_token> actual;
  int32_t drafted = 0;
  while (actual.size() < expected.size()) {
    llama_token candidate = LLAMA_TOKEN_NULL;
    assert(llama_dart_speculative_draft(
               speculative, 0, past, next, prompt.data(),
               static_cast<int32_t>(prompt.size()), 1, &candidate, 1) == 1);
    ++drafted;
    assert(llama_memory_seq_rm(
        llama_get_memory(llama_dart_speculative_get_draft_context(speculative)),
        0, past, -1));
    decode(target, {next, candidate}, past, speculative);
    const auto verified = greedy(target, 0);
    const bool accepted = candidate == verified;
    actual.push_back(next);
    prompt.push_back(next);
    if (accepted) {
      if (actual.size() < expected.size()) actual.push_back(candidate);
      prompt.push_back(candidate);
    }
    next = accepted ? greedy(target, 1) : verified;
    llama_dart_speculative_accept(speculative, 0, accepted ? 1 : 0);
    past += accepted ? 2 : 1;
    assert(llama_memory_seq_rm(llama_get_memory(target), 0, past, -1));
    assert(llama_memory_seq_rm(
        llama_get_memory(llama_dart_speculative_get_draft_context(speculative)),
        0, past, -1));
  }
  assert(drafted > 0);
  assert(actual == expected);
  std::puts("MTP greedy generation matches baseline after rollback probing");
}

static void test_mtp_rollback_probe(const char *model_path,
                                    const char *mtp_path) {
  llama_backend_init();
  auto model_params = llama_model_default_params();
  ggml_backend_dev_t devices[] = {nullptr};
  model_params.devices = devices;
  model_params.n_gpu_layers = 0;
  auto *model = llama_model_load_from_file(model_path, model_params);
  model_params.load_mtp = true;
  auto *draft_model = llama_model_load_from_file(mtp_path, model_params);
  assert(model != nullptr && draft_model != nullptr);
  auto params = llama_context_default_params();
  params.n_ctx = 256;
  params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  params.n_batch = 32;
  params.n_ubatch = 32;
  params.n_threads = params.n_threads_batch = 2;
  auto *target = llama_init_from_model(model, params);
  assert(target != nullptr);
  llama_dart_speculative_params speculative_params{};
  speculative_params.type_names = "draft-mtp";
  auto *speculative = llama_dart_speculative_init(model, draft_model, target,
                                                  params, &speculative_params);
  assert(speculative != nullptr);
  assert(llama_dart_speculative_get_draft_context(speculative) != nullptr);
  llama_dart_speculative_free(speculative);
  assert(llama_dart_mtp_target_is_empty(target));

  // A failed real decode rejects the pair. Abort each context in turn to
  // check cleanup and recovery, including shared target memory.
  const ggml_abort_callback abort = [](void *) { return true; };
  for (const bool abort_target : {true, false}) {
    llama_set_abort_callback(target, abort_target ? abort : nullptr, nullptr);
    params.abort_callback = abort_target ? nullptr : abort;
    speculative = llama_dart_speculative_init(model, draft_model, target,
                                              params, &speculative_params);
    assert(speculative == nullptr);
    assert(llama_dart_mtp_target_is_empty(target));
  }
  llama_set_abort_callback(target, nullptr, nullptr);
  params.abort_callback = nullptr;
  // Failure must not poison the target or prevent a later valid MTP init.
  speculative = llama_dart_speculative_init(model, draft_model, target, params,
                                            &speculative_params);
  assert(speculative != nullptr);
  test_mtp_generation(model, target, params, speculative);
  llama_dart_speculative_free(speculative);
  llama_free(target);
  llama_model_free(draft_model);
  llama_model_free(model);
  llama_backend_free();
  std::puts("MTP component rollback acceptance, per-context rejection, and "
            "recovery passed");
}

// Optional trained recurrent target coverage. The independent generated head
// isolates rollback classification from model-family/dimension compatibility.
static void test_mtp_recurrent_rollback_types(const char *target_path,
                                             const char *draft_path) {
  llama_backend_init();
  auto model_params = llama_model_default_params();
  ggml_backend_dev_t devices[] = {nullptr};
  model_params.devices = devices;
  model_params.n_gpu_layers = 0;
  auto *model = llama_model_load_from_file(target_path, model_params);
  model_params.load_mtp = true;
  auto *draft_model = llama_model_load_from_file(draft_path, model_params);
  assert(model != nullptr && draft_model != nullptr);
  for (const uint32_t snapshots : {0u, 2u}) {
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = params.n_ubatch = 8;
    params.n_threads = params.n_threads_batch = 2;
    params.n_rs_seq = snapshots;
    auto *target = llama_init_from_model(model, params);
    assert(target != nullptr);
    assert(llama_n_rs_seq(target) == snapshots);
    params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    params.ctx_other = target;
    params.n_rs_seq = 0;
    auto *draft = llama_init_from_model(draft_model, params);
    assert(draft != nullptr);
    const auto types = llama_dart_mtp_probe_rollback(target, draft);
    assert(types.first == (snapshots == 0 ? COMMON_CONTEXT_SEQ_RM_TYPE_FULL
                                        : COMMON_CONTEXT_SEQ_RM_TYPE_RS));
    assert(types.second == COMMON_CONTEXT_SEQ_RM_TYPE_PART);
    assert(llama_dart_mtp_target_is_empty(target));
    assert(llama_dart_mtp_target_is_empty(draft));
    llama_free(draft);
    llama_free(target);
  }
  llama_model_free(draft_model);
  llama_model_free(model);
  llama_backend_free();
  std::puts("Recurrent FULL/RS probe classification and cleanup passed (not "
            "MTP-family qualification)");
}

int main(int argc, char **argv) {
  if (argc == 4 && std::strcmp(argv[1], "--rollback-types") == 0) {
    test_mtp_recurrent_rollback_types(argv[2], argv[3]);
    return 0;
  }
  if (argc == 2 || argc == 3) {
#if defined(LLAMADART_TEST_CPU_BACKEND)
    assert(ggml_backend_load(LLAMADART_TEST_CPU_BACKEND) != nullptr);
#endif
    test_mtp_fresh_target_contract(argv[1], argc == 3 ? argv[2] : nullptr);
    if (argc == 3) {
      test_mtp_rollback_probe(argv[1], argv[2]);
    }
  } else {
    assert(argc == 1);
  }
  test_draft_moe_cache();
  test_nextn_query_export();
  assert(!llama_dart_speculative_need_embd_nextn(nullptr));

  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, true);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH, false);
  expect_requirements(COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK, false);

  const auto combined = llama_dart_speculative_embedding_requirements_for({
      COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
      COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
  });
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
  test_generic_accept_requires_draft();
  test_generic_draft_clamps_and_failed_draft();
  test_generic_repeated_accept_and_sequence_validation();
#if LLAMADART_SPECULATIVE_HAS_COMMON_BATCH
  test_generic_process_batch();
#endif
  return 0;
}
