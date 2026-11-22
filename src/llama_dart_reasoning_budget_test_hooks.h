#pragma once

#if defined(LLAMADART_TESTING)
#include "llama_dart_wrapper.h"

extern "C" LLAMADART_API void
llama_dart_test_set_reasoning_budget_init_hook(void (*hook)());
#endif
