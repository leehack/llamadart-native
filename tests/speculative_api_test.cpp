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
  return 0;
}
