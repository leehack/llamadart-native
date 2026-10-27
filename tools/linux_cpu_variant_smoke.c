// Loads a Linux bundle's CPU backends the way llamadart's auto mode does
// (ggml_backend_load_all_from_path), checks which libggml-cpu module ggml
// selected, and runs an F16 x F32 matmul on it. Under qemu-aarch64 -cpu, an
// instruction the emulated CPU lacks raises SIGILL, so a passing run shows that
// every variant loads and scores on that CPU and the selected one computes.
#define _GNU_SOURCE
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

enum { K = 256, N = 64, M = 8 };

static void print_features(ggml_backend_reg_t reg) {
  ggml_backend_get_features_t get_features =
      (ggml_backend_get_features_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
  if (get_features == NULL) {
    return;
  }
  printf("features:");
  for (struct ggml_backend_feature *feature = get_features(reg); feature->name != NULL; feature++) {
    printf(" %s=%s", feature->name, feature->value);
  }
  printf("\n");
}

static const char *selected_module(ggml_backend_reg_t reg) {
  Dl_info info;
  void *symbol = ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
  if (symbol == NULL || dladdr(symbol, &info) == 0 || info.dli_fname == NULL) {
    return NULL;
  }
  const char *slash = strrchr(info.dli_fname, '/');
  return slash != NULL ? slash + 1 : info.dli_fname;
}

static int run_f16_matmul(ggml_backend_dev_t device) {
  ggml_backend_t backend = ggml_backend_dev_init(device, NULL);
  if (backend == NULL) {
    fprintf(stderr, "ggml_backend_dev_init failed\n");
    return 1;
  }
  struct ggml_init_params params = {
      .mem_size = 4 * ggml_tensor_overhead() + ggml_graph_overhead(),
      .mem_buffer = NULL,
      .no_alloc = true,
  };
  struct ggml_context *ctx = ggml_init(params);
  struct ggml_tensor *a = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, K, N);
  struct ggml_tensor *b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);
  struct ggml_tensor *c = ggml_mul_mat(ctx, a, b);
  struct ggml_cgraph *graph = ggml_new_graph(ctx);
  ggml_build_forward_expand(graph, c);
  ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

  static float a32[N * K], b32[M * K], out[M * N];
  static ggml_fp16_t a16[N * K];
  for (int i = 0; i < N * K; i++) {
    a32[i] = (float)((i * 7) % 17 - 8) / 8.0f;
  }
  for (int i = 0; i < M * K; i++) {
    b32[i] = (float)((i * 5) % 13 - 6) / 4.0f;
  }
  ggml_fp32_to_fp16_row(a32, a16, N * K);
  ggml_fp16_to_fp32_row(a16, a32, N * K);
  ggml_backend_tensor_set(a, a16, 0, sizeof(a16));
  ggml_backend_tensor_set(b, b32, 0, sizeof(b32));

  int status = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS ? 0 : 1;
  ggml_backend_tensor_get(c, out, 0, sizeof(out));
  float max_error = 0.0f;
  for (int m = 0; m < M && status == 0; m++) {
    for (int n = 0; n < N; n++) {
      float expected = 0.0f;
      for (int k = 0; k < K; k++) {
        expected += a32[n * K + k] * b32[m * K + k];
      }
      float error = fabsf(out[m * N + n] - expected);
      max_error = error > max_error ? error : max_error;
    }
  }
  if (status != 0) {
    fprintf(stderr, "graph compute failed\n");
  } else if (max_error > 0.5f) {
    fprintf(stderr, "F16 matmul max error %g exceeds 0.5\n", max_error);
    status = 1;
  } else {
    printf("F16 matmul %dx%dx%d max error %g\n", M, N, K, max_error);
  }
  ggml_backend_buffer_free(buffer);
  ggml_free(ctx);
  ggml_backend_free(backend);
  return status;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <bundle-dir> <expected libggml-cpu module filename>\n", argv[0]);
    return 2;
  }
  ggml_backend_load_all_from_path(argv[1]);
  ggml_backend_dev_t device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  if (device == NULL) {
    fprintf(stderr, "No CPU backend loaded from %s\n", argv[1]);
    return 1;
  }
  ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
  const char *module = selected_module(reg);
  printf("selected: %s\n", module != NULL ? module : "(unknown)");
  print_features(reg);
  if (module == NULL || strcmp(module, argv[2]) != 0) {
    fprintf(stderr, "Expected %s to be selected\n", argv[2]);
    return 1;
  }
  return run_f16_matmul(device);
}
