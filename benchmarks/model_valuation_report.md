# Chimera-Inference: SOTA Model & Speculative Valuation Benchmark (Balanced Preset)

**Hardware Envelope**: NVIDIA GeForce RTX 3050 Laptop (4GB GDDR6 VRAM, 4095 MiB) | Intel Core i5-12500H (4 P-cores + 8 E-cores) | 16GB System RAM (WSL2 Ubuntu 24.04 LTS)

| Rank | Candidate / Model Pair | Intel Score | Speed (TPS) | VRAM (MiB) | Headroom | Value Score | Pareto? | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **#1** | `Qwen 2.5 1.5B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)` | 61.4 | 41.5 tok/s | 3030 | +1065 | **81.7** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#2** | `DeepSeek-R1-Distill-Qwen 1.5B + Qwen 2.5 0.5B (Intel P-cores AVX2)` | 66.5 | 38.6 tok/s | 3050 | +1045 | **80.8** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#3** | `Llama 3.2 3B Instruct + Llama 3.2 1B (Intel P-cores AVX2)` | 60.1 | 34.8 tok/s | 2550 | +1545 | **80.5** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#4** | `DeepSeek-R1-Distill-Qwen 1.5B + Prompt Lookup (Zero-FLOP)` | 66.5 | 35.8 tok/s | 3050 | +1045 | **76.5** | No | High-Synergy Speculative Champion |
| **#5** | `Qwen 2.5 3B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)` | 70.6 | 26.6 tok/s | 3650 | +445 | **76.0** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#6** | `Qwen 2.5 1.5B Instruct + Prompt Lookup (Zero-FLOP)` | 61.4 | 36.2 tok/s | 3030 | +1065 | **75.2** | No | High-Synergy Speculative Champion |
| **#7** | `Qwen 2.5 0.5B Instruct (Q4_K_M)` | 47.8 | 58.4 tok/s | 750 | +3345 | **73.6** | **Yes (SOTA)** | Viable Standalone |
| **#8** | `Llama 3.2 3B Instruct + Prompt Lookup (Zero-FLOP)` | 60.1 | 30.2 tok/s | 2550 | +1545 | **72.1** | No | High-Synergy Speculative Champion |
| **#9** | `DeepSeek-R1-Distill-Qwen 1.5B (Q4_K_M)` | 66.5 | 28.2 tok/s | 3050 | +1045 | **71.4** | No | Viable Standalone |
| **#10** | `Gemma 2 2B Instruct + Prompt Lookup (Zero-FLOP)` | 49.0 | 33.8 tok/s | 2150 | +1945 | **70.2** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#11** | `Qwen 2.5 1.5B Instruct (Q4_K_M)` | 61.4 | 28.4 tok/s | 3030 | +1065 | **69.8** | No | Viable Standalone |
| **#12** | `Llama 3.2 1B Instruct (Q4_K_M)` | 41.3 | 41.5 tok/s | 1250 | +2845 | **69.2** | No | Viable Standalone |
| **#13** | `Llama 3.2 3B Instruct (Q4_K_M)` | 60.1 | 23.8 tok/s | 2550 | +1545 | **67.1** | No | Viable Standalone |
| **#14** | `Gemma 2 2B Instruct (Q4_K_M)` | 49.0 | 27.0 tok/s | 2150 | +1945 | **65.5** | No | Viable Standalone |
| **#15** | `Qwen 2.5 3B Instruct (Q4_K_M)` | 70.6 | 17.5 tok/s | 3650 | +445 | **63.5** | No | Viable Standalone |
| **#16** | `Qwen 2.5 7B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)` | 81.6 | 10.7 tok/s | 4095 | +0 | **63.4** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#17** | `DeepSeek-R1-Distill-Qwen 7B + Qwen 2.5 0.5B (Intel P-cores AVX2)` | 83.3 | 10.2 tok/s | 4095 | +0 | **63.4** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#18** | `DeepSeek-R1-Distill-Llama 8B + Llama 3.2 1B (Intel P-cores AVX2)` | 77.0 | 8.5 tok/s | 4095 | +0 | **58.9** | No | High-Synergy Speculative Champion |
| **#19** | `K2-Horizon-7B + Prompt Lookup (Zero-FLOP)` | 85.3 | 7.1 tok/s | 4095 | +0 | **56.4** | **Yes (SOTA)** | High-Synergy Speculative Champion |
| **#20** | `DeepSeek-R1-Distill-Qwen 7B (Q4_K_M)` | 83.3 | 6.5 tok/s | 4095 | +0 | **53.7** | No | Constrained Standalone |
| **#21** | `Qwen 2.5 7B Instruct (Q4_K_M)` | 81.6 | 6.8 tok/s | 4095 | +0 | **53.5** | No | Constrained Standalone |
| **#22** | `K2-Horizon-7B (Q4_K_M)` | 85.3 | 5.4 tok/s | 4095 | +0 | **53.5** | No | Constrained Standalone |
| **#23** | `DeepSeek-R1-Distill-Llama 8B (Q4_K_M)` | 77.0 | 5.4 tok/s | 4095 | +0 | **50.6** | No | Constrained Standalone |
| **#24** | `Llama 3.1 8B Instruct (Q4_K_M)` | 75.6 | 5.6 tok/s | 4095 | +0 | **50.2** | No | Constrained Standalone |
| **#25** | `Llama 3.2 3B Instruct + Qwen 2.5 0.5B [MISMATCHED]` | 33.0 | 17.1 tok/s | 2550 | +1545 | **46.6** | No | Degraded / Toxic Speculative Pair |
| **#26** | `Llama 3.1 8B Instruct + TinyLlama 1.1B [UNDERPOWERED]` | 49.1 | 4.7 tok/s | 4095 | +0 | **33.1** | No | Degraded / Toxic Speculative Pair |
| **#27** | `K2-Horizon-7B + K2-Horizon 0.9B [MISMATCHED VOCAB]` | 46.9 | 3.8 tok/s | 4095 | +0 | **31.7** | No | Degraded / Toxic Speculative Pair |

### Pareto-Optimal Frontier Analysis

- **`Qwen 2.5 1.5B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)`** (Value: 81.7): TPS: 41.5, Intel: 61.4, VRAM: 3030 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (74% acceptance rate), High speculative acceleration (1.46x speedup -> 41.5 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`DeepSeek-R1-Distill-Qwen 1.5B + Qwen 2.5 0.5B (Intel P-cores AVX2)`** (Value: 80.8): TPS: 38.6, Intel: 66.5, VRAM: 3050 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (62% acceptance rate), High speculative acceleration (1.37x speedup -> 38.6 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`Llama 3.2 3B Instruct + Llama 3.2 1B (Intel P-cores AVX2)`** (Value: 80.5): TPS: 34.8, Intel: 60.1, VRAM: 2550 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (68% acceptance rate), High speculative acceleration (1.46x speedup -> 34.8 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`Qwen 2.5 3B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)`** (Value: 76.0): TPS: 26.6, Intel: 70.6, VRAM: 3650 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (71% acceptance rate), High speculative acceleration (1.52x speedup -> 26.6 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`Qwen 2.5 0.5B Instruct (Q4_K_M)`** (Value: 73.6): TPS: 58.4, Intel: 47.8, VRAM: 750 MiB. Viable Standalone. Advantages: Fluid generation (58.4 tok/s), Generous VRAM headroom (3345 MiB free)
- **`Gemma 2 2B Instruct + Prompt Lookup (Zero-FLOP)`** (Value: 70.2): TPS: 33.8, Intel: 49.0, VRAM: 2150 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (45% acceptance rate), High speculative acceleration (1.25x speedup -> 33.8 tok/s), Zero-FLOP, zero-VRAM Prompt Lookup synergy
- **`Qwen 2.5 7B Instruct + Qwen 2.5 0.5B (Intel P-cores AVX2)`** (Value: 63.4): TPS: 10.7, Intel: 81.6, VRAM: 4095 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (69% acceptance rate), High speculative acceleration (1.57x speedup -> 10.7 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`DeepSeek-R1-Distill-Qwen 7B + Qwen 2.5 0.5B (Intel P-cores AVX2)`** (Value: 63.4): TPS: 10.2, Intel: 83.3, VRAM: 4095 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (63% acceptance rate), High speculative acceleration (1.57x speedup -> 10.2 tok/s), CPU P-core drafting offloads compute from 4GB GPU
- **`K2-Horizon-7B + Prompt Lookup (Zero-FLOP)`** (Value: 56.4): TPS: 7.1, Intel: 85.3, VRAM: 4095 MiB. High-Synergy Speculative Champion. Advantages: Lossless distribution match (48% acceptance rate), High speculative acceleration (1.31x speedup -> 7.1 tok/s), Zero-FLOP, zero-VRAM Prompt Lookup synergy