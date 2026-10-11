#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
#include <omp.h>

#define CHIMERA_EXPORT extern "C" __attribute__((visibility("default")))

// Maximum supported P-cores on Intel i5-12500H
constexpr int MAX_P_CORES = 4;
constexpr int QK_K = 256;

struct TopologyInfo {
    int total_logical_cores;
    int detected_p_cores;
    int p_core_ids[MAX_P_CORES];
};

static TopologyInfo g_topology = {8, 4, {0, 1, 2, 3}};

CHIMERA_EXPORT void chimera_init_topology(int* out_p_cores, int* out_total_cores) {
    int total = omp_get_num_procs();
    g_topology.total_logical_cores = total;
    g_topology.detected_p_cores = (total >= 4) ? 4 : total;
    for (int i = 0; i < g_topology.detected_p_cores; ++i) {
        g_topology.p_core_ids[i] = i;
    }
    if (out_p_cores) *out_p_cores = g_topology.detected_p_cores;
    if (out_total_cores) *out_total_cores = g_topology.total_logical_cores;
}

CHIMERA_EXPORT int chimera_bind_current_thread_to_p_core(int p_core_index) {
    if (p_core_index < 0 || p_core_index >= g_topology.detected_p_cores) {
        return -1;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(g_topology.p_core_ids[p_core_index], &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}

// Half-precision float conversion via hardware F16C
static inline float fp16_to_fp32(uint16_t h) {
    return _cvtsh_ss(h);
}

// GGML Q4_K and Q6_K Super-block structures
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size must be 144 bytes");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size must be 210 bytes");

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

CHIMERA_EXPORT void chimera_dequantize_q4_k(const void* src, float* dst, int k) {
    dequantize_row_q4_K((const block_q4_K*)src, dst, k);
}

CHIMERA_EXPORT void chimera_dequantize_q6_k(const void* src, float* dst, int k) {
    dequantize_row_q6_K((const block_q6_K*)src, dst, k);
}

CHIMERA_EXPORT void chimera_dequantize_tensor_q4_k_p_cores(
    const void* __restrict src,
    float* __restrict dst,
    int n_elements
) {
    int n_blocks = n_elements / QK_K;
    const block_q4_K* blocks = (const block_q4_K*)src;
    int n_workers = g_topology.detected_p_cores;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        #pragma omp for schedule(static)
        for (int b = 0; b < n_blocks; ++b) {
            dequantize_row_q4_K(&blocks[b], dst + b * QK_K, QK_K);
        }
    }
}

CHIMERA_EXPORT void chimera_dequantize_tensor_q6_k_p_cores(
    const void* __restrict src,
    float* __restrict dst,
    int n_elements
) {
    int n_blocks = n_elements / QK_K;
    const block_q6_K* blocks = (const block_q6_K*)src;
    int n_workers = g_topology.detected_p_cores;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        #pragma omp for schedule(static)
        for (int b = 0; b < n_blocks; ++b) {
            dequantize_row_q6_K(&blocks[b], dst + b * QK_K, QK_K);
        }
    }
}

// AVX2 Dot Product Kernel (32 elements per iteration using 4 x 256-bit registers)
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

// P-Core Pinned Parallel GEMV: y = W * x
CHIMERA_EXPORT void chimera_gemv_p_cores(
    const float* __restrict W,
    const float* __restrict x,
    float* __restrict y,
    int rows,
    int cols
) {
    int n_workers = g_topology.detected_p_cores;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            y[r] = dot_product_avx2_unrolled(&W[r * cols], x, cols);
        }
    }
}

// P-Core Pinned Parallel Quantized GEMV for Q4_K: y = W_q4 * x
CHIMERA_EXPORT void chimera_gemv_q4_k_p_cores(
    const void* __restrict W_q4,
    const float* __restrict x,
    float* __restrict y,
    int rows,
    int cols
) {
    int bytes_per_row = (cols / QK_K) * sizeof(block_q4_K);
    int n_workers = g_topology.detected_p_cores;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        alignas(32) float local_row[16384];

        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const block_q4_K* blocks = (const block_q4_K*)((const uint8_t*)W_q4 + r * bytes_per_row);
            dequantize_row_q4_K(blocks, local_row, cols);
            y[r] = dot_product_avx2_unrolled(local_row, x, cols);
        }
    }
}

// P-Core Pinned Parallel Quantized GEMV for Q6_K: y = W_q6 * x
CHIMERA_EXPORT void chimera_gemv_q6_k_p_cores(
    const void* __restrict W_q6,
    const float* __restrict x,
    float* __restrict y,
    int rows,
    int cols
) {
    int bytes_per_row = (cols / QK_K) * sizeof(block_q6_K);
    int n_workers = g_topology.detected_p_cores;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

        alignas(32) float local_row[16384];

        #pragma omp for schedule(static)
        for (int r = 0; r < rows; ++r) {
            const block_q6_K* blocks = (const block_q6_K*)((const uint8_t*)W_q6 + r * bytes_per_row);
            dequantize_row_q6_K(blocks, local_row, cols);
            y[r] = dot_product_avx2_unrolled(local_row, x, cols);
        }
    }
}

// Standard Unpinned GEMV for direct A/B testing
CHIMERA_EXPORT void chimera_gemv_unpinned(
    const float* __restrict W,
    const float* __restrict x,
    float* __restrict y,
    int rows,
    int cols,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int r = 0; r < rows; ++r) {
        y[r] = dot_product_avx2_unrolled(&W[r * cols], x, cols);
    }
}

// AVX2 Accelerated RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
CHIMERA_EXPORT void chimera_rmsnorm(
    const float* __restrict x,
    const float* __restrict weight,
    float* __restrict out,
    int size,
    float eps
) {
    __m256 sum0 = _mm256_setzero_ps();
    __m256 sum1 = _mm256_setzero_ps();

    int i = 0;
    for (; i <= size - 16; i += 16) {
        __m256 v0 = _mm256_loadu_ps(x + i);
        __m256 v1 = _mm256_loadu_ps(x + i + 8);
        sum0 = _mm256_fmadd_ps(v0, v0, sum0);
        sum1 = _mm256_fmadd_ps(v1, v1, sum1);
    }
    sum0 = _mm256_add_ps(sum0, sum1);

    alignas(32) float buf[8];
    _mm256_store_ps(buf, sum0);
    float sum_sq = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];

    for (; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }

    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);
    __m256 v_inv_rms = _mm256_set1_ps(inv_rms);

    i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vw = _mm256_loadu_ps(weight + i);
        __m256 res = _mm256_mul_ps(_mm256_mul_ps(vx, v_inv_rms), vw);
        _mm256_storeu_ps(out + i, res);
    }
    for (; i < size; ++i) {
        out[i] = (x[i] * inv_rms) * weight[i];
    }
}

// Full-vocabulary P-core parallel argmax over Q6_K tied embeddings
CHIMERA_EXPORT void chimera_lm_head_greedy(
    const void* __restrict emb_buffer,
    const float* __restrict x,
    int vocab_size,
    int hidden_dim,
    int* out_best_id,
    float* out_best_logit
) {
    int bytes_per_token = (hidden_dim / QK_K) * sizeof(block_q6_K);
    int n_workers = (g_topology.detected_p_cores > 0) ? g_topology.detected_p_cores : 1;

    int global_best_id = -1;
    float global_best_logit = -1e30f;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        if (g_topology.detected_p_cores > 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }

        float local_row[8192];
        int local_best_id = -1;
        float local_best_logit = -1e30f;

        #pragma omp for schedule(static)
        for (int v = 0; v < vocab_size; ++v) {
            const block_q6_K* blocks = (const block_q6_K*)((const uint8_t*)emb_buffer + v * bytes_per_token);
            dequantize_row_q6_K(blocks, local_row, hidden_dim);
            float logit = dot_product_avx2_unrolled(local_row, x, hidden_dim);
            if (logit > local_best_logit) {
                local_best_logit = logit;
                local_best_id = v;
            }
        }

        #pragma omp critical
        {
            if (local_best_logit > global_best_logit) {
                global_best_logit = local_best_logit;
                global_best_id = local_best_id;
            }
        }
    }

    if (out_best_id) *out_best_id = global_best_id;
    if (out_best_logit) *out_best_logit = global_best_logit;
}

// Full-vocabulary P-core parallel logits calculation over Q6_K tied embeddings
CHIMERA_EXPORT void chimera_lm_head_logits(
    const void* __restrict emb_buffer,
    const float* __restrict x,
    int vocab_size,
    int hidden_dim,
    float* __restrict out_logits
) {
    int bytes_per_token = (hidden_dim / QK_K) * sizeof(block_q6_K);
    int n_workers = (g_topology.detected_p_cores > 0) ? g_topology.detected_p_cores : 1;

    #pragma omp parallel num_threads(n_workers) proc_bind(close)
    {
        int tid = omp_get_thread_num();
        if (g_topology.detected_p_cores > 0) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(g_topology.p_core_ids[tid % g_topology.detected_p_cores], &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }

        float local_row[8192];

        #pragma omp for schedule(static)
        for (int v = 0; v < vocab_size; ++v) {
            const block_q6_K* blocks = (const block_q6_K*)((const uint8_t*)emb_buffer + v * bytes_per_token);
            dequantize_row_q6_K(blocks, local_row, hidden_dim);
            out_logits[v] = dot_product_avx2_unrolled(local_row, x, hidden_dim);
        }
    }
}

