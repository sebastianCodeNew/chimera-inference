#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <cmath>
#include <pthread.h>
#include <sched.h>
#include <immintrin.h>
#include <omp.h>

// Simulated Qwen 2.5 Layer GEMV dimensions
constexpr int M_ROWS = 8960;  // FFN intermediate dimension
constexpr int K_COLS = 3584;  // Hidden dimension
constexpr int WARMUP_RUNS = 10;
constexpr int BENCH_RUNS = 40;

// AVX2 Single-Precision Fused Multiply-Add Dot Product
float avx2_dot_product(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    float buffer[8];
    _mm256_storeu_ps(buffer, sum);
    float total = 0.0f;
    for (int j = 0; j < 8; ++j) total += buffer[j];
    for (; i < n; ++i) total += a[i] * b[i];
    return total;
}

// Matrix-Vector Multiplication: y = W * x
void gemv_parallel(const float* W, const float* x, float* y, int rows, int cols, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int r = 0; r < rows; ++r) {
        y[r] = avx2_dot_product(&W[r * cols], x, cols);
    }
}

// Set Thread Affinity for OpenMP Team
void set_team_affinity(const std::vector<int>& core_ids) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid < static_cast<int>(core_ids.size())) {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_ids[tid], &cpuset);
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        }
    }
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "Chimera-Inference: Hardware Baseline CPU Benchmark" << std::endl;
    std::cout << "Matrix Dimensions: [" << M_ROWS << " x " << K_COLS << "] (FFN GEMV)" << std::endl;
    std::cout << "Max OpenMP Threads Available: " << omp_get_max_threads() << std::endl;
    std::cout << "==================================================" << std::endl;

    std::vector<float> W(M_ROWS * K_COLS, 0.05f);
    std::vector<float> x(K_COLS, 1.0f);
    std::vector<float> y(M_ROWS, 0.0f);

    // Warmup
    gemv_parallel(W.data(), x.data(), y.data(), M_ROWS, K_COLS, omp_get_max_threads());

    struct Config {
        std::string name;
        int threads;
        std::vector<int> core_ids;
    };

    std::vector<Config> configs = {
        {"1. All Logical Cores (Unpinned - llama.cpp default)", omp_get_max_threads(), {}},
        {"2. 4 P-Cores Physical (Cores 0, 1, 2, 3 - Pinned)", 4, {0, 1, 2, 3}},
        {"3. 4 P-Cores Hyperthreaded (8 Threads 0-7 - Pinned)", 8, {0, 1, 2, 3, 4, 5, 6, 7}}
    };

    for (const auto& cfg : configs) {
        if (!cfg.core_ids.empty()) {
            set_team_affinity(cfg.core_ids);
        }

        std::vector<double> latencies_ms;
        latencies_ms.reserve(BENCH_RUNS);

        for (int r = 0; r < BENCH_RUNS; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            gemv_parallel(W.data(), x.data(), y.data(), M_ROWS, K_COLS, cfg.threads);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            latencies_ms.push_back(ms);
        }

        double sum = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
        double mean = sum / BENCH_RUNS;

        double sq_sum = 0.0;
        for (double val : latencies_ms) sq_sum += (val - mean) * (val - mean);
        double stdev = std::sqrt(sq_sum / BENCH_RUNS);

        double gflops = (2.0 * M_ROWS * K_COLS * 1e-9) / (mean * 1e-3);
        double gbps = (static_cast<double>(M_ROWS) * K_COLS * 4.0 * 1e-9) / (mean * 1e-3);

        std::cout << "\nConfig: " << cfg.name << std::endl;
        std::cout << "  Mean Latency:    " << mean << " ms (+/- " << stdev << " ms)" << std::endl;
        std::cout << "  Compute Rate:    " << gflops << " GFLOPS" << std::endl;
        std::cout << "  Memory Read Rate: " << gbps << " GB/s" << std::endl;
    }
    std::cout << "\n==================================================" << std::endl;
    return 0;
}
