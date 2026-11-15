#include "llama_dart_wrapper.h"
#include "mtmd-helper.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <string>

#ifdef NDEBUG
#error "ABI canaries require active assertions"
#endif

namespace {
// Test/handoff admission harness. The consumer must gate its own first FFI
// struct call. This owner test cannot change an already published consumer.
bool admits(const char *identity) {
  if (!identity) return false;
  const std::string value(identity);
  if (value.rfind("llamadart-native-build/1;", 0) != 0) return false;
  const std::string key = ";llama.cpp=";
  const auto at = value.find(key);
  if (at == std::string::npos || value.find(key, at + 1) != std::string::npos) return false;
  const auto end = value.find(';', at + key.size());
  if (end == std::string::npos) return false;
  const auto source = value.substr(at + key.size(), end - at - key.size());
#if LLAMADART_CONTEXT_HAS_MOE_CPU_SHARE
  return source == "b1b770d90d94cad9e9ea5b993bdbc383a522fb28";
#else
  return source == "cd959bae99ff4bed22bef2604f45241c09bbb8c8" ||
         source == "8642541b255bfb5dc39cd351a178be09013a55d6";
#endif
}

template<class Defaults>
bool admitted_defaults(const char *identity, Defaults defaults) {
  if (!admits(identity)) return false;
  defaults();
  return true;
}
}

int main() {
  int calls = 0;
  const auto spy = [&] { ++calls; };
  for (const char *identity : {static_cast<const char *>(nullptr), "", "unknown",
      "llamadart-native-build/2;llama.cpp=b1b770d90d94cad9e9ea5b993bdbc383a522fb28;",
      "llamadart-native-build/1;llama.cpp=unknown;",
      "llamadart-native-build/1;llama.cpp=b1b770d90d94cad9e9ea5b993bdbc383a522fb28+dirty.123;",
      "llamadart-native-build/1;llama.cpp=b1b770d90d94cad9e9ea5b993bdbc383a522fb28;llama.cpp=unknown;"}) {
    assert(!admitted_defaults(identity, spy));
  }
#if LLAMADART_CONTEXT_HAS_MOE_CPU_SHARE
  assert(!admitted_defaults("llamadart-native-build/1;llama.cpp=cd959bae99ff4bed22bef2604f45241c09bbb8c8;", spy));
#else
  assert(!admitted_defaults("llamadart-native-build/1;llama.cpp=b1b770d90d94cad9e9ea5b993bdbc383a522fb28;", spy));
#endif
  assert(calls == 0); // No by-value call made on any refused runtime.
  assert(admitted_defaults(llama_dart_build_identity(), [&] {
    ++calls;
    const auto model = llama_model_default_params();
    const auto context = llama_context_default_params();
    assert(!model.load_mtp && !model.vocab_only);
    assert(context.moe_cache_size == 0 && context.ctx_other == nullptr);
    assert(!context.abort_callback && !context.abort_callback_data);
    assert(!context.samplers && context.n_samplers == 0);
    assert(context.type_k == GGML_TYPE_F16 && context.type_v == GGML_TYPE_F16);
#if LLAMADART_CONTEXT_HAS_MOE_CPU_SHARE
    assert(context.moe_cache_cpu_share == 0.0f);
#else
    assert(context.defrag_thold == -1.0f);
#endif
  }));
  assert(calls == 1);
  // All currently shipped native ABIs are 64-bit. Do not silently bless an
  // unreviewed 32-bit ABI from host-only size assertions.
  static_assert(sizeof(void *) == 8, "review layout for this ABI");
  static_assert(alignof(llama_context_params) == 8, "context alignment changed");
  static_assert(sizeof(llama_model_params) == 80);
#if LLAMADART_MTMD_AUDIO_HAS_TEMP
  static_assert(sizeof(mtmd_helper_gen_audio_inp) == 72);
  static_assert(alignof(mtmd_helper_gen_audio_inp) == 8);
  static_assert(offsetof(mtmd_helper_gen_audio_inp, temp) == 56);
  static_assert(offsetof(mtmd_helper_gen_audio_inp, seed) == 60);
  assert(mtmd_helper_gen_audio_inp_default().temp == -1.0f);
#endif
#if LLAMADART_CONTEXT_HAS_MOE_CPU_SHARE
  static_assert(sizeof(llama_context_params) == 176);
  static_assert(offsetof(llama_context_params, moe_cache_cpu_share) == 120);
  static_assert(offsetof(llama_context_params, abort_callback) == 128);
  static_assert(offsetof(llama_context_params, samplers) == 152);
  static_assert(offsetof(llama_context_params, ctx_other) == 168);
  static_assert(GGML_OP_COUNT == 99);
#else
  static_assert(sizeof(llama_context_params) == 168);
  static_assert(offsetof(llama_context_params, abort_callback) == 120);
  static_assert(offsetof(llama_context_params, samplers) == 144);
  static_assert(offsetof(llama_context_params, ctx_other) == 160);
  // v0.8 and v0.9 use the pre-pruning enum.
  static_assert(GGML_OP_COUNT == 102);
#endif
  std::printf("context: sizeof=%zu alignof=%zu moe_cache=%zu abort=%zu samplers=%zu ctx_other=%zu; model=%zu; audio_input=%zu; ggml_ops=%d\n",
      sizeof(llama_context_params), alignof(llama_context_params),
      offsetof(llama_context_params, moe_cache_size), offsetof(llama_context_params, abort_callback),
      offsetof(llama_context_params, samplers), offsetof(llama_context_params, ctx_other),
      sizeof(llama_model_params), sizeof(mtmd_helper_gen_audio_inp), GGML_OP_COUNT);
}
