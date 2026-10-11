#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
#include <omp.h>

constexpr int QK_K = 256;

static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

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

    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

int main() {
    int vocab_size = 151936;
    int hidden_dim = 1536;
    int blocks_per_token = hidden_dim / QK_K; // 6
    int bytes_per_token = blocks_per_token * sizeof(block_q6_K); // 1260

    const char* path = "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4";
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return 1;

    // Seek to tensor_data_offset (token_embd.weight is at offset 0)
    int64_t tensor_data_offset = 5950976;
    f.seekg(tensor_data_offset);

    size_t total_emb_bytes = (size_t)vocab_size * bytes_per_token;
    std::vector<uint8_t> emb_buffer(total_emb_bytes);
    f.read((char*)emb_buffer.data(), total_emb_bytes);
    std::cout << "Read " << total_emb_bytes / (1024*1024) << " MB embedding weights into RAM." << std::endl;

    std::vector<float> x(hidden_dim, 0.05f);

    auto t0 = std::chrono::high_resolution_clock::now();
    int best_id = -1;
    float best_logit = -1e9f;

    #pragma omp parallel num_threads(4)
    {
        float local_row[1536];
        int local_best_id = -1;
        float local_best_logit = -1e9f;

        #pragma omp for schedule(static)
        for (int v = 0; v < vocab_size; ++v) {
            const block_q6_K* blocks = (const block_q6_K*)(emb_buffer.data() + v * bytes_per_token);
            dequantize_row_q6_K(blocks, local_row, hidden_dim);
            float logit = dot_product_avx2_unrolled(local_row, x.data(), hidden_dim);
            if (logit > local_best_logit) {
                local_best_logit = logit;
                local_best_id = v;
            }
        }

        #pragma omp critical
        {
            if (local_best_logit > best_logit) {
                best_logit = local_best_logit;
                best_id = local_best_id;
            }
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Computed argmax over all 151,936 vocab tokens in " << ms << " ms!" << std::endl;
    std::cout << "Best token ID: " << best_id << ", logit: " << best_logit << std::endl;
    return 0;
}
