#pragma once

#include "llama_dart_barrier_internal.h"
#include "llama_dart_wrapper.h"

// A backend can still be computing when evaluation returns or throws. Keep
// its wait inside the call in flight; preserve an evaluation exception even
// when that wait also throws. Inject the guard and wait for deterministic
// contract tests, as for the barrier's other internal helpers.
template <typename Guard, typename Context, typename Call, typename Synchronize>
static int32_t llama_dart_evaluate(Context *ctx, Call &&call,
                                  Synchronize &&synchronize) {
  return llama_dart_barrier<int32_t>(
      LLAMA_DART_STATUS_EXCEPTION, [ctx, &call, &synchronize]() -> int32_t {
        Guard in_flight;
        int32_t status;
        try {
          status = call();
        } catch (...) {
          try {
            synchronize(ctx);
          } catch (...) {
          }
          throw;
        }
        synchronize(ctx);
        return status;
      });
}
