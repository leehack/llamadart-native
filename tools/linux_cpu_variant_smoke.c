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
      if (!isfinite(out[m * N + n]) || !isfinite(error)) {
        fprintf(stderr, "F16 matmul produced a nonfinite output/error at row %d col %d\n", n, m);
        status = 1;
        break;
      }
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

static int run_finite_dot_cases(ggml_backend_dev_t device) {
  ggml_backend_t backend = ggml_backend_dev_init(device, NULL);
  if (backend == NULL) {
    fprintf(stderr, "ggml_backend_dev_init failed\n");
    return 1;
  }
  ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
  ggml_backend_set_n_threads_t set_threads =
      (ggml_backend_set_n_threads_t)ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
  if (set_threads == NULL) {
    fprintf(stderr, "CPU thread setter unavailable\n");
    ggml_backend_free(backend);
    return 1;
  }
  set_threads(backend, 1);
  int status = 0;
  for (int test_case = 0; test_case < 11 && status == 0; test_case++) {
    const int f32_control = test_case == 10;
    const int cancellation = test_case == 9;
    const float magnitude = test_case < 8 ? 30.0f + 10.0f * test_case : test_case == 8 ? 1.0f : 100.0f;
    struct ggml_init_params params = {
        .mem_size = 4 * ggml_tensor_overhead() + ggml_graph_overhead(),
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    struct ggml_context *ctx = ggml_init(params);
    if (ctx == NULL) {
      fprintf(stderr, "finite dot context allocation failed\n");
      status = 1;
      break;
    }
    struct ggml_tensor *k = ggml_new_tensor_2d(ctx, f32_control ? GGML_TYPE_F32 : GGML_TYPE_F16, 64, 1);
    struct ggml_tensor *q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 1);
    struct ggml_tensor *out = ggml_mul_mat(ctx, k, q);
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == NULL) {
      fprintf(stderr, "finite dot tensor buffer allocation failed\n");
      ggml_free(ctx);
      status = 1;
      break;
    }
    ggml_fp16_t keys16[64];
    float keys32[64], queries[64];
    double reference = 0.0, absolute_sum = 0.0;
    for (int i = 0; i < 64; i++) {
      keys32[i] = cancellation ? magnitude : magnitude * (1.0f + 0.1f * (i % 5));
      queries[i] = cancellation ? (i < 32 ? magnitude : -magnitude)
                                : magnitude * (1.0f + 0.1f * ((i * 3) % 5));
      keys16[i] = ggml_fp32_to_fp16(keys32[i]);
      const double actual_k = f32_control ? keys32[i] : ggml_fp16_to_fp32(keys16[i]);
      const double actual_q = f32_control ? queries[i] : ggml_fp16_to_fp32(ggml_fp32_to_fp16(queries[i]));
      const double product = actual_k * actual_q;
      reference += product;
      absolute_sum += fabs(product);
    }
    ggml_backend_tensor_set(k, f32_control ? (void *)keys32 : (void *)keys16, 0,
                            f32_control ? sizeof(keys32) : sizeof(keys16));
    ggml_backend_tensor_set(q, queries, 0, sizeof(queries));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
      fprintf(stderr, "finite dot graph computation failed\n");
      status = 1;
    } else {
      float actual;
      ggml_backend_tensor_get(out, &actual, 0, sizeof(actual));
      const double error = fabs((double)actual - reference);
      if (!isfinite(actual) || !isfinite(error) || error > 1e-7 + 2e-6 * absolute_sum) {
        fprintf(stderr, "finite dot case %d mag %.0f expected %.17g got %.9g\n",
                test_case, magnitude, reference, actual);
        status = 1;
      }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
  }
  ggml_backend_free(backend);
  if (status == 0) {
    printf("finite F16 dot: magnitude sweep, low/cancellation/F32 controls passed\n");
  }
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
  int status = run_f16_matmul(device);
  return status != 0 ? status : run_finite_dot_cases(device);
}
