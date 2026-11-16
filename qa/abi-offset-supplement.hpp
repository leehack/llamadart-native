#include <cstddef>
#include "llama.h"
static_assert(offsetof(llama_context_params, moe_cache_size) == 112,
              "candidate MoE cache offset changed");
