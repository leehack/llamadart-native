// Execute against the real compiled CPU backend, never a copied selector.
// Synthetic capability masks only inspect tables; they must not run kernels.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-feats.h"
#include "kernels.h"
#include "kleidiai.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static int expected_mask(int mask, ggml_type type) {
    if (type == GGML_TYPE_Q4_0) {
        if ((mask & 48) == 48) return 48; // SME2 + FP16
#ifndef __APPLE__
        if ((mask & 7) == 7) return 7; // SVE + I8MM + DOTPROD
        if ((mask & 3) == 3) return 3;
#endif
        if (mask & 1) return 1;
    } else if (type == GGML_TYPE_Q8_0) {
        if (mask & 16) return 16;
        if (mask & 8) return 8;
        if ((mask & 3) == 3) return 3;
        if (mask & 1) return 1;
    } else if (type == GGML_TYPE_F32) {
        if (mask & 16) return 16;
        if (mask & 8) return 8;
    } else if (type == GGML_TYPE_F16 && (mask & 16)) {
        return 16;
    }
    return -1;
}

static void check_selection(ggml_kleidiai_kernels * kernel, int expected, int mask) {
    require((kernel != nullptr) == (expected >= 0), "selector null/fallback mismatch");
    if (kernel) {
        require(static_cast<int>(kernel->required_cpu) == expected, "selector chose wrong capability family");
        require((mask & kernel->required_cpu) == kernel->required_cpu, "selector bypassed required feature");
        require(kernel->gemm.run_kernel_ex && kernel->rhs_info.pack_func_ex, "selected incomplete kernel");
    }
}

static void selector_tests() {
    const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q5_0 };
    for (int bits = 0; bits < 64; ++bits) {
        auto mask = static_cast<cpu_feature>(bits);
        check_selection(ggml_kleidiai_select_kernels_q4_0(mask), expected_mask(bits, GGML_TYPE_Q4_0), bits);
        check_selection(ggml_kleidiai_select_kernels_q8_0(mask), expected_mask(bits, GGML_TYPE_Q8_0), bits);
        check_selection(ggml_kleidiai_select_kernels_f32(mask), expected_mask(bits, GGML_TYPE_F32), bits);
        for (auto type : types) {
            ggml_tensor weights = {}, input = {}, output = {};
            weights.type = type;
            input.type = output.type = GGML_TYPE_F32;
            output.op = GGML_OP_MUL_MAT;
            output.src[0] = &weights;
            output.src[1] = &input;
            check_selection(ggml_kleidiai_select_kernels(mask, &output), expected_mask(bits, type), bits);
            output.op = GGML_OP_ADD;
            require(!ggml_kleidiai_select_kernels(mask, &output), "wrong operation accepted");
            output.op = GGML_OP_MUL_MAT;
            output.src[1] = nullptr;
            require(!ggml_kleidiai_select_kernels(mask, &output), "missing source accepted");
            output.src[1] = &input;
            output.src[0] = nullptr;
            require(!ggml_kleidiai_select_kernels(mask, &output), "missing weights accepted");
            output.src[0] = &weights;
            input.type = GGML_TYPE_I32;
            require(!ggml_kleidiai_select_kernels(mask, &output), "wrong input type accepted");
            input.type = GGML_TYPE_F32;
            output.type = GGML_TYPE_F16;
            require(!ggml_kleidiai_select_kernels(mask, &output), "wrong output type accepted");
        }
    }
    std::puts("PASS: real selectors, all 64 feature masks, positive and negative tensor paths");
}

static void matmul_test(ggml_type type, int columns, bool use_kleidiai) {
    constexpr int k = 64, rows = 16;
    auto backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend unavailable");
    ggml_backend_cpu_set_n_threads(backend, 2);
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    auto weights_ctx = ggml_init(params);
    auto ctx = ggml_init(params);
    require(weights_ctx && ctx, "context allocation failed");
    auto weights = ggml_new_tensor_2d(weights_ctx, type, k, rows);
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, columns);
    auto output = ggml_mul_mat(ctx, weights, input);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    // Explicitly use the production KleidiAI repacking buffer, so this cannot
    // pass by exercising only the ordinary GGML fallback implementation.
    auto weights_buffer = ggml_backend_alloc_ctx_tensors_from_buft(weights_ctx,
        use_kleidiai ? ggml_backend_cpu_kleidiai_buffer_type() : ggml_backend_cpu_buffer_type());
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(weights_buffer && buffer, "CPU buffer unavailable");
    require((weights->extra != nullptr) == use_kleidiai, "wrong optimized/fallback buffer path");
    std::vector<float> raw(k * rows), restored(k * rows), inputs(k * columns);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = (static_cast<int>(i * 13 % 17) - 8) * 0.125f;
    for (size_t i = 0; i < inputs.size(); ++i) inputs[i] = (i * 7 % 11 < 5) ? -1.0f : 1.0f;
    std::vector<unsigned char> quantized(ggml_nbytes(weights));
    require(ggml_quantize_chunk(type, raw.data(), quantized.data(), 0, rows, k, nullptr) == quantized.size(),
            "quantization size mismatch");
    ggml_get_type_traits(type)->to_float(quantized.data(), restored.data(), restored.size());
    ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "KleidiAI graph compute failed");
    std::vector<float> actual(rows * columns);
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (int n = 0; n < columns; ++n) {
        for (int m = 0; m < rows; ++m) {
            float expected = 0;
            for (int i = 0; i < k; ++i) expected += restored[m * k + i] * inputs[n * k + i];
            if (!std::isfinite(actual[n * rows + m]) || std::fabs(actual[n * rows + m] - expected) > 0.005f) {
                std::fprintf(stderr, "%s columns=%d row=%d col=%d expected=%f actual=%f\n",
                             ggml_type_name(type), columns, m, n, expected, actual[n * rows + m]);
                require(false, "KleidiAI result differs from scalar dequantized reference");
            }
        }
    }
    std::printf("PASS: %s %s matmul %dx%dx%d vs scalar reference\n", ggml_type_name(type),
                use_kleidiai ? "KleidiAI repack +" : "baseline fallback", rows, columns, k);
    ggml_backend_buffer_free(buffer);
    ggml_backend_buffer_free(weights_buffer);
    ggml_free(ctx);
    ggml_free(weights_ctx);
    ggml_backend_free(backend);
}

int main(int argc, char ** argv) {
    require(argc == 1 || (argc == 2 && std::strcmp(argv[1], "--selectors-only") == 0), "unknown test arguments");
    selector_tests();
    // Synthetic masks must never cause advanced kernels to execute.
    if (argc > 1) return 0;
    const auto runtime = ggml_feats_get_arch64_runtime();
    std::printf("Runtime: dotprod=%d i8mm=%d sve=%d sme=%d sme2=%d\n", runtime.has_dotprod,
                runtime.has_i8mm, runtime.has_sve, runtime.has_sme, runtime.has_sme2);
    // Both Q4_0 and Q8_0 have NEON dot-product implementations. Without it,
    // exercise the ordinary CPU fallback, never force unsupported kernels.
    const bool use_kleidiai = runtime.has_dotprod;
    for (auto type : { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0 }) {
        for (int columns : { 1, 4, 9 }) matmul_test(type, columns, use_kleidiai);
    }
    return 0;
}
