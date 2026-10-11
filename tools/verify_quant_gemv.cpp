#include <iostream>
#include <vector>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <random>
#include <immintrin.h>
#include <omp.h>
#include <sched.h>
#include <pthread.h>

constexpr int QK_K = 256;

static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline void dequantize_row_q4_K(const block_q4_K* x, float* y, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >>  4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static inline void dequantize_row_q6_K(const block_q6_K* x, float* y, int k) {
    int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

static inline float dot_product_avx2_unrolled(const float* __restrict a, const float* __restrict b, int n) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();
    __m256 sum2 = _mm256_setzero_ps();
    __m256 sum3 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= n - 32; i += 32) {
        sum0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      sum0);
        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  sum1);
        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), sum2);
        sum3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), sum3);
    }
    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    alignas(32) float buf[8];
    _mm256_store_ps(buf, sum0);
    float total = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

void gemv_q4_k(const void* W, const float* x, float* y, int rows, int cols) {
    int bytes_per_row = (cols / QK_K) * sizeof(block_q4_K);
    #pragma omp parallel num_threads(4) proc_bind(close)
    {
        alignas(32) float local_row[16384];
        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const block_q4_K* blocks = (const block_q4_K*)((const uint8_t*)W + r * bytes_per_row);
            dequantize_row_q4_K(blocks, local_row, cols);
            y[r] = dot_product_avx2_unrolled(local_row, x, cols);
        }
    }
}

void gemv_q6_k(const void* W, const float* x, float* y, int rows, int cols) {
    int bytes_per_row = (cols / QK_K) * sizeof(block_q6_K);
    #pragma omp parallel num_threads(4) proc_bind(close)
    {
        alignas(32) float local_row[16384];
        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const block_q6_K* blocks = (const block_q6_K*)((const uint8_t*)W + r * bytes_per_row);
            dequantize_row_q6_K(blocks, local_row, cols);
            y[r] = dot_product_avx2_unrolled(local_row, x, cols);
        }
    }
}

int main() {
    std::mt19937 rng(42);
    int rows = 512, cols = 1024;
    
    // Test Q4_K
    {
        int bytes = rows * (cols / QK_K) * sizeof(block_q4_K);
        std::vector<uint8_t> W(bytes);
        for (auto& b : W) b = rng() & 0xFF;
        std::vector<float> x(cols);
        for (auto& v : x) v = (rng() % 100) / 50.0f - 1.0f;
        
        std::vector<float> y1(rows, 0.0f);
        gemv_q4_k(W.data(), x.data(), y1.data(), rows, cols);
        
        // Baseline: full dequantize then gemv
        std::vector<float> W_f32(rows * cols);
        int nb = (cols / QK_K);
        for (int r = 0; r < rows; ++r) {
            const block_q4_K* b = (const block_q4_K*)(W.data() + r * nb * sizeof(block_q4_K));
            dequantize_row_q4_K(b, &W_f32[r * cols], cols);
        }
        std::vector<float> y2(rows, 0.0f);
        for (int r = 0; r < rows; ++r) {
            y2[r] = dot_product_avx2_unrolled(&W_f32[r * cols], x.data(), cols);
        }
        
        float max_diff = 0.0f;
        for (int r = 0; r < rows; ++r) {
            max_diff = std::max(max_diff, std::abs(y1[r] - y2[r]));
        }
        std::cout << "Q4_K max absolute difference: " << max_diff << std::endl;
        assert(max_diff == 0.0f);
    }

    // Test Q6_K
    {
        int bytes = rows * (cols / QK_K) * sizeof(block_q6_K);
        std::vector<uint8_t> W(bytes);
        for (auto& b : W) b = rng() & 0xFF;
        std::vector<float> x(cols);
        for (auto& v : x) v = (rng() % 100) / 50.0f - 1.0f;
        
        std::vector<float> y1(rows, 0.0f);
        gemv_q6_k(W.data(), x.data(), y1.data(), rows, cols);
        
        std::vector<float> W_f32(rows * cols);
        int nb = (cols / QK_K);
        for (int r = 0; r < rows; ++r) {
            const block_q6_K* b = (const block_q6_K*)(W.data() + r * nb * sizeof(block_q6_K));
            dequantize_row_q6_K(b, &W_f32[r * cols], cols);
        }
        std::vector<float> y2(rows, 0.0f);
        for (int r = 0; r < rows; ++r) {
            y2[r] = dot_product_avx2_unrolled(&W_f32[r * cols], x.data(), cols);
        }
        
        float max_diff = 0.0f;
        for (int r = 0; r < rows; ++r) {
            max_diff = std::max(max_diff, std::abs(y1[r] - y2[r]));
        }
        std::cout << "Q6_K max absolute difference: " << max_diff << std::endl;
        assert(max_diff == 0.0f);
    }
    
    std::cout << "ALL NUMERICAL CHECKS PASSED BIT-FOR-BIT!" << std::endl;
    return 0;
}
