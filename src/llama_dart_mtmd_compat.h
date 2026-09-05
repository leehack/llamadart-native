#pragma once

#include "mtmd-helper.h"

// v0.4.0 added helper options; retain builds against the published v0.3.0
// headers and use upstream's defaults when the options API is available.
static inline mtmd_helper_bitmap_wrapper llama_dart_bitmap_from_buffer(
    mtmd_context *ctx, const unsigned char *buffer, size_t size) {
#if LLAMADART_MTMD_HELPER_HAS_OPTIONS
  return mtmd_helper_bitmap_init_from_buf(
      ctx, buffer, size, false, mtmd_helper_init_opt_default());
#else
  return mtmd_helper_bitmap_init_from_buf(ctx, buffer, size, false);
#endif
}
