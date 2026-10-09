// libllamadart behind functions that tests/manual/dart_exit_probe.dart can
// bind without the structures of llama.h.
#include "llama.h"
#include <dlfcn.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
extern "C" {
llama_model *llama_dart_model_load_from_file(const char *, llama_model_params);
llama_context *llama_dart_init_from_model(llama_model *, llama_context_params);
int32_t llama_dart_decode(llama_context *, llama_batch);
llama_token llama_dart_sampler_sample(llama_sampler *, llama_context *, int32_t);
void llama_dart_set_log_level(int);
}
static void *g_lib;
template <typename F> static F *sym(const char *n) {
  void *s = dlsym(g_lib, n);
  if (!s) { fprintf(stderr, "missing %s\n", n); _Exit(3); }
  return reinterpret_cast<F *>(s);
}
#define BIND(name) auto *name##_ = sym<decltype(name)>(#name)
struct shim_session { llama_model *model; llama_context *ctx; llama_sampler *sampler; std::vector<llama_token> tokens; };
extern "C" {
int shim_open_library(const char *bundle) {
  if (g_lib) return 0;
  g_lib = dlopen((std::string(bundle) + "/libllamadart.so").c_str(), RTLD_LAZY);
  if (!g_lib) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  BIND(llama_dart_set_log_level); BIND(llama_backend_init); BIND(ggml_backend_load_all_from_path);
  llama_dart_set_log_level_(4); llama_backend_init_(); ggml_backend_load_all_from_path_(bundle);
  return 0;
}
void *shim_open_session(const char *model, int tokens) {
  BIND(llama_model_default_params); BIND(llama_dart_model_load_from_file);
  BIND(llama_context_default_params); BIND(llama_dart_init_from_model);
  BIND(llama_sampler_chain_default_params); BIND(llama_sampler_chain_init);
  BIND(llama_sampler_chain_add); BIND(llama_sampler_init_greedy);
  auto *s = new shim_session;
  auto mp = llama_model_default_params_();
  const char *g = getenv("PROBE_GPU_LAYERS");
  mp.n_gpu_layers = g ? atoi(g) : 0;
  s->model = llama_dart_model_load_from_file_(model, mp);
  if (!s->model) return nullptr;
  auto cp = llama_context_default_params_();
  cp.n_ctx = 2048; cp.n_batch = 2048;
  s->ctx = llama_dart_init_from_model_(s->model, cp);
  if (!s->ctx) return nullptr;
  s->sampler = llama_sampler_chain_init_(llama_sampler_chain_default_params_());
  llama_sampler_chain_add_(s->sampler, llama_sampler_init_greedy_());
  for (int i = 0; i < tokens; ++i) s->tokens.push_back(100 + i % 400);
  return s;
}
// One generation step: guarded decode, then guarded sample.
int shim_step(void *session) {
  BIND(llama_get_memory); BIND(llama_memory_clear); BIND(llama_dart_decode);
  BIND(llama_dart_sampler_sample); BIND(llama_batch_get_one);
  auto *s = static_cast<shim_session *>(session);
  llama_memory_clear_(llama_get_memory_(s->ctx), true);
  int status = llama_dart_decode_(s->ctx, llama_batch_get_one_(s->tokens.data(), (int32_t)s->tokens.size()));
  if (status != 0) return status;
  llama_dart_sampler_sample_(s->sampler, s->ctx, -1);
  return 0;
}
void shim_sleeping_atexit(int ms) {
  static int delay; delay = ms;
  atexit([] { std::this_thread::sleep_for(std::chrono::milliseconds(delay)); });
}
}
