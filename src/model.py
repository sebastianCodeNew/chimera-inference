"""Heterogeneous Transformer Architecture for Chimera-Inference Engine.

Orchestrates RMSNorm, RoPE rotary position embeddings, Multi-Head / Grouped
Query Attention (GQA), and SwiGLU FFN across Intel P-core AVX2 kernels and
NVIDIA Ampere CUDA / Tensor Core acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    HAS_TORCH = False

from src.core_engine import ChimeraCoreEngine
from src.gguf_parser import GGUFParser
from src.kv_cache import KVCache


@dataclass(frozen=True)
class TransformerConfig:
    """Architectural hyperparameters for Qwen 2.5 models."""

    vocab_size: int = 151936
    hidden_dim: int = 1536
    intermediate_dim: int = 8960
    n_layers: int = 28
    n_heads: int = 12
    n_kv_heads: int = 2
    head_dim: int = 128
    rope_base: float = 1000000.0
    rms_norm_eps: float = 1e-6
    max_seq_len: int = 2048


def softmax(
    x: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Numerically stable softmax."""
    exp_x = np.exp(x - np.max(x), dtype=np.float32)
    return (exp_x / np.sum(exp_x)).astype(np.float32)


def silu(
    x: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """SiLU (Swish) activation function: x / (1 + exp(-x))."""
    return (x / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)


def apply_rotary_emb(
    x: np.ndarray[Any, np.dtype[np.float32]],
    pos: int,
    head_dim: int = 128,
    base: float = 1000000.0,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Apply NeoX-style split-half Rotary Positional Embeddings (RoPE) on CPU."""
    half_dim = head_dim // 2
    inv_freq = 1.0 / (base ** (np.arange(0, half_dim, dtype=np.float32) / half_dim))
    theta = pos * inv_freq
    cos = np.cos(theta, dtype=np.float32)
    sin = np.sin(theta, dtype=np.float32)
    cos_full = np.concatenate([cos, cos]).astype(np.float32)
    sin_full = np.concatenate([sin, sin]).astype(np.float32)

    # Split-half rotation [-x2, x1]
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated_half = np.concatenate([-x2, x1], axis=-1)

    return (x * cos_full + rotated_half * sin_full).astype(np.float32)


def apply_rotary_emb_torch(
    x: Any,
    pos: int,
    head_dim: int = 128,
    base: float = 1000000.0,
    cos_table: Any = None,
    sin_table: Any = None,
) -> Any:
    """Apply NeoX-style split-half Rotary Positional Embeddings (RoPE) on GPU."""
    half_dim = head_dim // 2
    if cos_table is not None and sin_table is not None:
        cos = cos_table[pos]
        sin = sin_table[pos]
    else:
        device = x.device
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim
            )
        )
        theta = pos * inv_freq
        cos = torch.cos(theta).to(x.dtype)
        sin = torch.sin(theta).to(x.dtype)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    rotated_half = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated_half * sin


def rmsnorm_torch(
    x: Any,
    weight: Any,
    eps: float = 1e-6,
) -> Any:
    """Numerically stable RMSNorm on GPU."""
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps).to(x.dtype)) * weight


def gemv_quant(
    engine: ChimeraCoreEngine,
    W: np.ndarray[Any, Any],
    x: np.ndarray[Any, np.dtype[np.float32]],
    rows: int,
    cols: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Automatically dispatch to Q4_K, Q6_K, or FP32 P-core GEMV kernels."""
    if W.dtype == np.uint8:
        q4_size = rows * (cols // 256) * 144
        if W.nbytes == q4_size:
            return engine.gemv_q4_k_p_cores(W, x, rows, cols)
        return engine.gemv_q6_k_p_cores(W, x, rows, cols)
    return engine.gemv_p_cores(W, x)


class TransformerLayer:
    """Single Transformer Decoder Layer with GQA & SwiGLU across CPU/CUDA."""

    def __init__(
        self,
        layer_idx: int,
        config: TransformerConfig,
        attn_norm: Any,
        w_q: Any,
        b_q: Any,
        w_k: Any,
        b_k: Any,
        w_v: Any,
        b_v: Any,
        w_out: Any,
        ffn_norm: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        device: str = "cpu",
    ) -> None:
        self.layer_idx = layer_idx
        self.config = config
        self.device = device.lower()

        if self.device == "cuda":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is required for CUDA execution")
            torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            def to_cuda_fp16(t: Any) -> Any:
                if isinstance(t, torch.Tensor):
                    return t.to(device=torch_dev, dtype=torch.float16)
                return torch.as_tensor(t, device=torch_dev, dtype=torch.float16)

            self.attn_norm = to_cuda_fp16(attn_norm)
            self.w_q = to_cuda_fp16(w_q)
            self.b_q = to_cuda_fp16(b_q)
            self.w_k = to_cuda_fp16(w_k)
            self.b_k = to_cuda_fp16(b_k)
            self.w_v = to_cuda_fp16(w_v)
            self.b_v = to_cuda_fp16(b_v)
            self.w_out = to_cuda_fp16(w_out)
            self.ffn_norm = to_cuda_fp16(ffn_norm)
            self.w_gate = to_cuda_fp16(w_gate)
            self.w_up = to_cuda_fp16(w_up)
            self.w_down = to_cuda_fp16(w_down)

            # Precompute RoPE tables on CUDA
            half_dim = config.head_dim // 2
            inv_freq = 1.0 / (
                config.rope_base
                ** (
                    torch.arange(0, half_dim, dtype=torch.float32, device=torch_dev)
                    / half_dim
                )
            )
            positions = torch.arange(
                config.max_seq_len, dtype=torch.float32, device=torch_dev
            )
            theta = torch.outer(positions, inv_freq)
            cos = torch.cos(theta).to(torch.float16)
            sin = torch.sin(theta).to(torch.float16)
            self.cos_table: Any = torch.cat([cos, cos], dim=-1)
            self.sin_table: Any = torch.cat([sin, sin], dim=-1)
        else:
            self.attn_norm = attn_norm
            self.w_q = w_q
            self.b_q = b_q
            self.w_k = w_k
            self.b_k = b_k
            self.w_v = w_v
            self.b_v = b_v
            self.w_out = w_out
            self.ffn_norm = ffn_norm
            self.w_gate = w_gate
            self.w_up = w_up
            self.w_down = w_down
            self.cos_table = None
            self.sin_table = None

        self.scale = np.float32(1.0 / np.sqrt(config.head_dim))
        self.q_heads_per_kv = config.n_heads // config.n_kv_heads

    def forward(
        self,
        x: Any,
        pos: int,
        kv_cache: KVCache,
        engine: ChimeraCoreEngine,
    ) -> Any:
        """Compute single-token layer forward pass."""
        if self.device == "cuda":
            return self._forward_cuda(x, pos, kv_cache)
        return self._forward_cpu(x, pos, kv_cache, engine)

    def _forward_cuda(
        self,
        x: Any,
        pos: int,
        kv_cache: KVCache,
    ) -> Any:
        """Compute single-token layer forward pass using CUDA Tensor Cores."""
        # 1. Attention RMSNorm
        x_norm = rmsnorm_torch(x, self.attn_norm, eps=self.config.rms_norm_eps)

        # 2. Q, K, V Projections
        q = F.linear(x_norm, self.w_q, self.b_q).view(
            self.config.n_heads, self.config.head_dim
        )
        k = F.linear(x_norm, self.w_k, self.b_k).view(
            self.config.n_kv_heads, self.config.head_dim
        )
        v = F.linear(x_norm, self.w_v, self.b_v).view(
            self.config.n_kv_heads, self.config.head_dim
        )

        # 3. Reshape and RoPE with precomputed tables
        q = apply_rotary_emb_torch(
            q,
            pos,
            self.config.head_dim,
            self.config.rope_base,
            self.cos_table,
            self.sin_table,
        )
        k = apply_rotary_emb_torch(
            k,
            pos,
            self.config.head_dim,
            self.config.rope_base,
            self.cos_table,
            self.sin_table,
        )

        # 4. KV Cache Update
        k_cached, v_cached = kv_cache.update(self.layer_idx, k, v, pos)

        # 5. Grouped Query Attention (GQA) via hardware-accelerated SDPA
        q_sdpa = q.unsqueeze(0).unsqueeze(2)
        k_sdpa = (
            k_cached.to(torch.float16)
            .permute(1, 0, 2)
            .repeat_interleave(self.q_heads_per_kv, dim=0)
            .unsqueeze(0)
        )
        v_sdpa = (
            v_cached.permute(1, 0, 2)
            .repeat_interleave(self.q_heads_per_kv, dim=0)
            .unsqueeze(0)
        )

        attn_out = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa).reshape(
            self.config.hidden_dim
        )
        attn_proj = F.linear(attn_out, self.w_out)

        # 6. Residual Connection
        x = x + attn_proj

        # 7. FFN RMSNorm
        x_ffn_norm = rmsnorm_torch(x, self.ffn_norm, eps=self.config.rms_norm_eps)

        # 8. SwiGLU Feed-Forward Network
        gate = F.linear(x_ffn_norm, self.w_gate)
        up = F.linear(x_ffn_norm, self.w_up)
        ffn_hidden = (F.silu(gate.float()) * up.float()).to(torch.float16)
        ffn_proj = F.linear(ffn_hidden, self.w_down)

        # 9. Residual Connection
        x = x + ffn_proj
        return x

    def _forward_cpu(
        self,
        x: np.ndarray[Any, np.dtype[np.float32]],
        pos: int,
        kv_cache: KVCache,
        engine: ChimeraCoreEngine,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Compute single-token layer forward pass using P-core GEMV."""
        # 1. Attention RMSNorm
        x_norm = engine.rmsnorm(x, self.attn_norm, eps=self.config.rms_norm_eps)

        # 2. Q, K, V Projections
        q = (
            gemv_quant(
                engine,
                self.w_q,
                x_norm,
                self.config.n_heads * self.config.head_dim,
                self.config.hidden_dim,
            )
            + self.b_q
        )
        k = (
            gemv_quant(
                engine,
                self.w_k,
                x_norm,
                self.config.n_kv_heads * self.config.head_dim,
                self.config.hidden_dim,
            )
            + self.b_k
        )
        v = (
            gemv_quant(
                engine,
                self.w_v,
                x_norm,
                self.config.n_kv_heads * self.config.head_dim,
                self.config.hidden_dim,
            )
            + self.b_v
        )

        # 3. Reshape and RoPE
        q = q.reshape(self.config.n_heads, self.config.head_dim)
        k = k.reshape(self.config.n_kv_heads, self.config.head_dim)
        v = v.reshape(self.config.n_kv_heads, self.config.head_dim)

        q = apply_rotary_emb(q, pos, self.config.head_dim, self.config.rope_base)
        k = apply_rotary_emb(k, pos, self.config.head_dim, self.config.rope_base)

        # 4. KV Cache Update
        k_cached, v_cached = kv_cache.update(self.layer_idx, k, v, pos)

        # 5. Grouped Query Attention (GQA)
        head_outputs: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for h in range(self.config.n_heads):
            kv_h = h // self.q_heads_per_kv
            q_h = q[h]
            k_h = k_cached[:, kv_h, :]  # shape: (seq_len, head_dim)
            v_h = v_cached[:, kv_h, :]  # shape: (seq_len, head_dim)

            scores = (np.dot(k_h, q_h) * self.scale).astype(np.float32)
            probs = softmax(scores)
            head_out = np.dot(probs, v_h).astype(np.float32)
            head_outputs.append(head_out)

        attn_concat = np.concatenate(head_outputs, axis=0).astype(np.float32)
        attn_proj = gemv_quant(
            engine,
            self.w_out,
            attn_concat,
            self.config.hidden_dim,
            self.config.n_heads * self.config.head_dim,
        )

        # 6. Residual Connection
        x = x + attn_proj

        # 7. FFN RMSNorm
        x_ffn_norm = engine.rmsnorm(x, self.ffn_norm, eps=self.config.rms_norm_eps)

        # 8. SwiGLU Feed-Forward Network
        gate = gemv_quant(
            engine,
            self.w_gate,
            x_ffn_norm,
            self.config.intermediate_dim,
            self.config.hidden_dim,
        )
        up = gemv_quant(
            engine,
            self.w_up,
            x_ffn_norm,
            self.config.intermediate_dim,
            self.config.hidden_dim,
        )
        ffn_hidden = (silu(gate) * up).astype(np.float32)
        ffn_proj = gemv_quant(
            engine,
            self.w_down,
            ffn_hidden,
            self.config.hidden_dim,
            self.config.intermediate_dim,
        )

        # 9. Residual Connection
        x = x + ffn_proj
        return x


class ChimeraModel:
    """Full Transformer Generative Model with Heterogeneous CPU/CUDA Acceleration."""

    def __init__(
        self,
        config: TransformerConfig,
        engine: ChimeraCoreEngine,
        output_norm: Any,
        emb_buffer: Any,
        parser: GGUFParser | None = None,
        layers: list[TransformerLayer] | None = None,
        max_cached_layers: int = 28,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.engine = engine
        self.parser = parser
        self.max_cached_layers = max_cached_layers
        self.device = device.lower()

        if self.device == "cuda":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is required for CUDA execution")
            self.torch_device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
            if isinstance(output_norm, torch.Tensor):
                self.output_norm = output_norm.to(
                    device=self.torch_device, dtype=torch.float16
                )
            else:
                self.output_norm = torch.as_tensor(
                    output_norm, device=self.torch_device, dtype=torch.float16
                )

            if isinstance(emb_buffer, torch.Tensor):
                self.emb_cuda = emb_buffer.to(
                    device=self.torch_device, dtype=torch.float16
                )
            elif isinstance(emb_buffer, np.ndarray):
                self.emb_cuda = torch.from_numpy(emb_buffer).to(
                    device=self.torch_device, dtype=torch.float16
                )
            else:
                self.emb_cuda = None  # Will be managed via parser or mock
            self.emb_buffer = emb_buffer

            # Precompute RoPE tables for prompt prefill
            half_dim = config.head_dim // 2
            inv_freq = 1.0 / (
                config.rope_base
                ** (
                    torch.arange(
                        0, half_dim, dtype=torch.float32, device=self.torch_device
                    )
                    / half_dim
                )
            )
            positions = torch.arange(
                config.max_seq_len, dtype=torch.float32, device=self.torch_device
            )
            theta = torch.outer(positions, inv_freq)
            cos = torch.cos(theta).to(torch.float16)
            sin = torch.sin(theta).to(torch.float16)
            self.cos_table: Any = torch.cat([cos, cos], dim=-1)
            self.sin_table: Any = torch.cat([sin, sin], dim=-1)
        else:
            self.output_norm = output_norm
            self.emb_buffer = emb_buffer
            self.emb_cuda = None
            self.torch_device = None
            self.cos_table = None
            self.sin_table = None

        self._layers: dict[int, TransformerLayer] = {}
        if layers:
            for lyr in layers:
                self._layers[lyr.layer_idx] = lyr

        self.bytes_per_token_emb = (config.hidden_dim // 256) * 210

    @classmethod
    def from_gguf(
        cls,
        gguf_source: Path | str | GGUFParser,
        engine: ChimeraCoreEngine | None = None,
        n_layers: int | None = None,
        max_cached_layers: int = 28,
        preload_layers: bool = True,
        device: str = "cpu",
    ) -> Self:
        """Instantiate ChimeraModel directly from GGUF model blob."""
        if engine is None:
            engine = ChimeraCoreEngine()

        if isinstance(gguf_source, (str, Path)):
            parser = GGUFParser(gguf_source)
        else:
            parser = gguf_source

        config = TransformerConfig(
            vocab_size=int(parser.metadata.get("tokenizer.ggml.tokens_count", 151936)),
            hidden_dim=parser.embedding_length or 1536,
            intermediate_dim=int(
                parser.metadata.get("qwen2.feed_forward_length", 8960)
            ),
            n_layers=n_layers or parser.block_count or 28,
            n_heads=parser.head_count or 12,
            n_kv_heads=parser.head_count_kv or 2,
            head_dim=128,
            rope_base=float(parser.metadata.get("qwen2.rope.freq_base", 1000000.0)),
            rms_norm_eps=float(
                parser.metadata.get("qwen2.attention.layer_norm_rms_epsilon", 1e-6)
            ),
        )

        dev_norm = device.lower()
        if dev_norm == "cuda":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is required for CUDA execution")
            torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            out_norm_np = parser.load_tensor(
                "output_norm.weight", engine, dequantize=True
            )
            output_norm = torch.from_numpy(out_norm_np).to(
                device=torch_dev, dtype=torch.float16
            )

            emb_np = parser.load_tensor("token_embd.weight", engine, dequantize=True)
            emb_buffer = torch.from_numpy(emb_np).to(
                device=torch_dev, dtype=torch.float16
            )
        else:
            output_norm = parser.load_tensor("output_norm.weight", engine)
            emb_info = parser.tensors["token_embd.weight"]
            bytes_per_token = (config.hidden_dim // 256) * 210
            total_emb_bytes = config.vocab_size * bytes_per_token

            with open(parser.file_path, "rb") as f:
                f.seek(parser.tensor_data_offset + emb_info.offset)
                emb_buffer = f.read(total_emb_bytes)

        model = cls(
            config=config,
            engine=engine,
            output_norm=output_norm,
            emb_buffer=emb_buffer,
            parser=parser,
            max_cached_layers=max_cached_layers,
            device=device,
        )

        # Preload layers into memory to avoid per-token disk I/O
        if preload_layers:
            for l_idx in range(config.n_layers):
                model.get_layer(l_idx)
        else:
            model.get_layer(0)
        return model

    @classmethod
    def create_mock(
        cls,
        config: TransformerConfig | None = None,
        device: str = "cpu",
    ) -> Self:
        """Create lightweight mock model for fast deterministic testing."""
        if config is None:
            config = TransformerConfig(
                vocab_size=2048,
                hidden_dim=64,
                intermediate_dim=128,
                n_layers=2,
                n_heads=4,
                n_kv_heads=2,
                head_dim=16,
            )

        engine = ChimeraCoreEngine()
        dev_norm = device.lower()

        if dev_norm == "cuda":
            if not HAS_TORCH:
                raise RuntimeError("PyTorch is required for CUDA execution")
            torch_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(42)

            output_norm = torch.ones(
                config.hidden_dim, device=torch_dev, dtype=torch.float16
            )
            emb_buffer = (
                torch.randn(
                    config.vocab_size,
                    config.hidden_dim,
                    device=torch_dev,
                    dtype=torch.float16,
                )
                * 0.05
            )

            layers: list[TransformerLayer] = []
            for l_idx in range(config.n_layers):
                layer = TransformerLayer(
                    layer_idx=l_idx,
                    config=config,
                    attn_norm=torch.ones(
                        config.hidden_dim, device=torch_dev, dtype=torch.float16
                    ),
                    w_q=torch.eye(
                        config.hidden_dim, device=torch_dev, dtype=torch.float16
                    ),
                    b_q=torch.zeros(
                        config.hidden_dim, device=torch_dev, dtype=torch.float16
                    ),
                    w_k=torch.eye(
                        config.n_kv_heads * config.head_dim,
                        config.hidden_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    b_k=torch.zeros(
                        config.n_kv_heads * config.head_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    w_v=torch.eye(
                        config.n_kv_heads * config.head_dim,
                        config.hidden_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    b_v=torch.zeros(
                        config.n_kv_heads * config.head_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    w_out=torch.eye(
                        config.hidden_dim, device=torch_dev, dtype=torch.float16
                    ),
                    ffn_norm=torch.ones(
                        config.hidden_dim, device=torch_dev, dtype=torch.float16
                    ),
                    w_gate=torch.eye(
                        config.intermediate_dim,
                        config.hidden_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    w_up=torch.eye(
                        config.intermediate_dim,
                        config.hidden_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    w_down=torch.eye(
                        config.hidden_dim,
                        config.intermediate_dim,
                        device=torch_dev,
                        dtype=torch.float16,
                    ),
                    device=device,
                )
                layers.append(layer)

            return cls(
                config=config,
                engine=engine,
                output_norm=output_norm,
                emb_buffer=emb_buffer,
                parser=None,
                layers=layers,
                device=device,
            )

        np.random.seed(42)
        output_norm = np.ones(config.hidden_dim, dtype=np.float32)
        bytes_per_token = (config.hidden_dim // 256 + 1) * 210
        emb_buffer = bytearray(config.vocab_size * bytes_per_token)

        layers_cpu: list[TransformerLayer] = []
        for l_idx in range(config.n_layers):
            layer = TransformerLayer(
                layer_idx=l_idx,
                config=config,
                attn_norm=np.ones(config.hidden_dim, dtype=np.float32),
                w_q=np.eye(config.hidden_dim, dtype=np.float32),
                b_q=np.zeros(config.hidden_dim, dtype=np.float32),
                w_k=np.eye(
                    config.n_kv_heads * config.head_dim,
                    config.hidden_dim,
                    dtype=np.float32,
                ),
                b_k=np.zeros(config.n_kv_heads * config.head_dim, dtype=np.float32),
                w_v=np.eye(
                    config.n_kv_heads * config.head_dim,
                    config.hidden_dim,
                    dtype=np.float32,
                ),
                b_v=np.zeros(config.n_kv_heads * config.head_dim, dtype=np.float32),
                w_out=np.eye(config.hidden_dim, dtype=np.float32),
                ffn_norm=np.ones(config.hidden_dim, dtype=np.float32),
                w_gate=np.eye(
                    config.intermediate_dim, config.hidden_dim, dtype=np.float32
                ),
                w_up=np.eye(
                    config.intermediate_dim, config.hidden_dim, dtype=np.float32
                ),
                w_down=np.eye(
                    config.hidden_dim, config.intermediate_dim, dtype=np.float32
                ),
                device=device,
            )
            layers_cpu.append(layer)

        return cls(
            config=config,
            engine=engine,
            output_norm=output_norm,
            emb_buffer=emb_buffer,
            parser=None,
            layers=layers_cpu,
            device=device,
        )

    def get_layer(self, layer_idx: int) -> TransformerLayer:
        """Fetch layer weights with LRU workspace caching."""
        if layer_idx in self._layers:
            return self._layers[layer_idx]

        if self.parser is None:
            raise RuntimeError(
                f"Layer {layer_idx} not found and no GGUF parser attached"
            )

        # Evict oldest if exceeding max_cached_layers
        if len(self._layers) >= self.max_cached_layers:
            oldest_key = next(iter(self._layers))
            del self._layers[oldest_key]

        prefix = f"blk.{layer_idx}."
        if self.device == "cuda":
            layer = TransformerLayer(
                layer_idx=layer_idx,
                config=self.config,
                attn_norm=self.parser.load_tensor(
                    f"{prefix}attn_norm.weight", self.engine, dequantize=True
                ),
                w_q=self.parser.load_tensor(
                    f"{prefix}attn_q.weight", self.engine, dequantize=True
                ),
                b_q=self.parser.load_tensor(
                    f"{prefix}attn_q.bias", self.engine, dequantize=True
                ),
                w_k=self.parser.load_tensor(
                    f"{prefix}attn_k.weight", self.engine, dequantize=True
                ),
                b_k=self.parser.load_tensor(
                    f"{prefix}attn_k.bias", self.engine, dequantize=True
                ),
                w_v=self.parser.load_tensor(
                    f"{prefix}attn_v.weight", self.engine, dequantize=True
                ),
                b_v=self.parser.load_tensor(
                    f"{prefix}attn_v.bias", self.engine, dequantize=True
                ),
                w_out=self.parser.load_tensor(
                    f"{prefix}attn_output.weight", self.engine, dequantize=True
                ),
                ffn_norm=self.parser.load_tensor(
                    f"{prefix}ffn_norm.weight", self.engine, dequantize=True
                ),
                w_gate=self.parser.load_tensor(
                    f"{prefix}ffn_gate.weight", self.engine, dequantize=True
                ),
                w_up=self.parser.load_tensor(
                    f"{prefix}ffn_up.weight", self.engine, dequantize=True
                ),
                w_down=self.parser.load_tensor(
                    f"{prefix}ffn_down.weight", self.engine, dequantize=True
                ),
                device=self.device,
            )
        else:
            layer = TransformerLayer(
                layer_idx=layer_idx,
                config=self.config,
                attn_norm=self.parser.load_tensor(
                    f"{prefix}attn_norm.weight", self.engine
                ),
                w_q=self.parser.load_tensor(
                    f"{prefix}attn_q.weight", self.engine, dequantize=False
                ),
                b_q=self.parser.load_tensor(f"{prefix}attn_q.bias", self.engine),
                w_k=self.parser.load_tensor(
                    f"{prefix}attn_k.weight", self.engine, dequantize=False
                ),
                b_k=self.parser.load_tensor(f"{prefix}attn_k.bias", self.engine),
                w_v=self.parser.load_tensor(
                    f"{prefix}attn_v.weight", self.engine, dequantize=False
                ),
                b_v=self.parser.load_tensor(f"{prefix}attn_v.bias", self.engine),
                w_out=self.parser.load_tensor(
                    f"{prefix}attn_output.weight", self.engine, dequantize=False
                ),
                ffn_norm=self.parser.load_tensor(
                    f"{prefix}ffn_norm.weight", self.engine
                ),
                w_gate=self.parser.load_tensor(
                    f"{prefix}ffn_gate.weight", self.engine, dequantize=False
                ),
                w_up=self.parser.load_tensor(
                    f"{prefix}ffn_up.weight", self.engine, dequantize=False
                ),
                w_down=self.parser.load_tensor(
                    f"{prefix}ffn_down.weight", self.engine, dequantize=False
                ),
                device=self.device,
            )

        self._layers[layer_idx] = layer
        return layer

    def get_token_embedding(
        self, token_id: int
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        """Extract and dequantize token embedding vector across CPU & CUDA."""
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise IndexError(f"Token ID {token_id} out of vocabulary range")

        if self.device == "cuda" and self.emb_cuda is not None:
            return self.emb_cuda[token_id].cpu().float().numpy().copy()

        if self.parser is None:
            np.random.seed(token_id + 1)
            return (
                np.random.randn(self.config.hidden_dim).astype(np.float32) * 0.05
            ).copy()

        offset = token_id * self.bytes_per_token_emb
        raw = self.emb_buffer[offset : offset + self.bytes_per_token_emb]
        return self.engine.dequantize_q6_k(raw, self.config.hidden_dim)

    def forward_token(
        self,
        token_id: int,
        pos: int,
        kv_cache: KVCache,
        compute_lm_head: bool = True,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Compute full forward pass for a single token and return greedy next token."""
        if self.device == "cuda":
            return self._forward_token_cuda(
                token_id, pos, kv_cache, compute_lm_head, return_logits=return_logits
            )
        return self._forward_token_cpu(
            token_id, pos, kv_cache, compute_lm_head, return_logits=return_logits
        )

    def forward_prompt(
        self,
        prompt_token_ids: list[int],
        kv_cache: KVCache,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Compute parallel batched prompt prefill across all prompt tokens."""
        if not prompt_token_ids:
            raise ValueError("Cannot prefill empty prompt")
        if self.device == "cuda":
            return self._forward_prompt_cuda(
                prompt_token_ids, kv_cache, return_logits=return_logits
            )
        return self._forward_prompt_cpu(
            prompt_token_ids, kv_cache, return_logits=return_logits
        )

    def forward_token_logits(
        self,
        token_id: int,
        pos: int,
        kv_cache: KVCache,
    ) -> tuple[int, float, Any]:
        """Forward a token and return (best_id, best_logit, logits)."""
        res = self.forward_token(
            token_id, pos, kv_cache, compute_lm_head=True, return_logits=True
        )
        return res  # type: ignore[return-value]

    def forward_prompt_logits(
        self,
        prompt_token_ids: list[int],
        kv_cache: KVCache,
    ) -> tuple[int, float, Any]:
        """Forward prompt and return (best_id, best_logit, logits)."""
        res = self.forward_prompt(prompt_token_ids, kv_cache, return_logits=True)
        return res  # type: ignore[return-value]

    def _forward_prompt_cuda(
        self,
        prompt_token_ids: list[int],
        kv_cache: KVCache,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Compute parallel batched prompt prefill on CUDA with Tensor Cores."""
        t_len = len(prompt_token_ids)
        if t_len > self.config.max_seq_len:
            raise ValueError(
                f"Prompt length {t_len} exceeds max sequence length {self.config.max_seq_len}"
            )

        for tid in prompt_token_ids:
            if tid < 0 or tid >= self.config.vocab_size:
                raise IndexError(f"Token ID {tid} out of vocabulary range")

        with torch.inference_mode():
            if self.emb_cuda is not None:
                x = self.emb_cuda[prompt_token_ids]
            else:
                embs = [self.get_token_embedding(t) for t in prompt_token_ids]
                x = torch.as_tensor(
                    np.stack(embs, axis=0),
                    device=self.torch_device,
                    dtype=torch.float16,
                )

            cos_t = self.cos_table[:t_len].unsqueeze(1)
            sin_t = self.sin_table[:t_len].unsqueeze(1)
            half_dim = self.config.head_dim // 2
            q_heads_per_kv = self.config.n_heads // self.config.n_kv_heads

            for l_idx in range(self.config.n_layers):
                layer = self.get_layer(l_idx)
                # 1. Attn norm
                x_norm = rmsnorm_torch(x, layer.attn_norm, eps=self.config.rms_norm_eps)

                # 2. QKV projections
                q = F.linear(x_norm, layer.w_q, layer.b_q).view(
                    t_len, self.config.n_heads, self.config.head_dim
                )
                k = F.linear(x_norm, layer.w_k, layer.b_k).view(
                    t_len, self.config.n_kv_heads, self.config.head_dim
                )
                v = F.linear(x_norm, layer.w_v, layer.b_v).view(
                    t_len, self.config.n_kv_heads, self.config.head_dim
                )

                # 3. RoPE
                q1, q2 = q[..., :half_dim], q[..., half_dim:]
                q = q * cos_t + torch.cat([-q2, q1], dim=-1) * sin_t

                k1, k2 = k[..., :half_dim], k[..., half_dim:]
                k = k * cos_t + torch.cat([-k2, k1], dim=-1) * sin_t

                # 4. Update KV cache
                kv_cache.update_prompt(l_idx, k, v, t_len)

                # 5. Causal SDPA
                q_sdpa = q.permute(1, 0, 2).unsqueeze(0)
                k_sdpa = (
                    k.repeat_interleave(q_heads_per_kv, dim=1)
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                )
                v_sdpa = (
                    v.repeat_interleave(q_heads_per_kv, dim=1)
                    .permute(1, 0, 2)
                    .unsqueeze(0)
                )

                attn_out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa, is_causal=True
                )
                attn_out = (
                    attn_out.squeeze(0)
                    .permute(1, 0, 2)
                    .reshape(t_len, self.config.hidden_dim)
                )
                attn_proj = F.linear(attn_out, layer.w_out)
                x = x + attn_proj

                # 6. FFN RMSNorm & SwiGLU
                x_ffn_norm = rmsnorm_torch(
                    x, layer.ffn_norm, eps=self.config.rms_norm_eps
                )
                gate = F.linear(x_ffn_norm, layer.w_gate)
                up = F.linear(x_ffn_norm, layer.w_up)
                ffn_hidden = (F.silu(gate.float()) * up.float()).to(torch.float16)
                ffn_proj = F.linear(ffn_hidden, layer.w_down)
                x = x + ffn_proj

            # 7. Output RMSNorm and LM Head for last prompt token
            last_x = x[-1]
            last_norm = rmsnorm_torch(
                last_x, self.output_norm, eps=self.config.rms_norm_eps
            )

            if self.emb_cuda is not None:
                logits = F.linear(last_norm, self.emb_cuda)
                best_logit, best_id = torch.max(logits, dim=-1)
                if return_logits:
                    return int(best_id.item()), float(best_logit.item()), logits
                return int(best_id.item()), float(best_logit.item())

            best_id = (prompt_token_ids[-1] + 1) % self.config.vocab_size
            best_logit = float(torch.dot(last_norm, last_norm).item())
            if return_logits:
                mock_logits = torch.zeros(
                    self.config.vocab_size,
                    device=self.torch_device,
                    dtype=torch.float32,
                )
                mock_logits[best_id] = best_logit
                return best_id, best_logit, mock_logits
            return best_id, best_logit

    def _forward_prompt_cpu(
        self,
        prompt_token_ids: list[int],
        kv_cache: KVCache,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Sequential prompt prefill on CPU."""
        for pos, tid in enumerate(prompt_token_ids[:-1]):
            self.forward_token(tid, pos, kv_cache, compute_lm_head=False)
        last_pos = len(prompt_token_ids) - 1
        last_tid = prompt_token_ids[-1]
        return self.forward_token(
            last_tid,
            last_pos,
            kv_cache,
            compute_lm_head=True,
            return_logits=return_logits,
        )

    def _forward_token_cuda(
        self,
        token_id: int,
        pos: int,
        kv_cache: KVCache,
        compute_lm_head: bool = True,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Compute full forward pass for a single token on CUDA."""
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise IndexError(f"Token ID {token_id} out of vocabulary range")

        with torch.inference_mode():
            if self.emb_cuda is not None:
                x = self.emb_cuda[token_id]
            else:
                x = torch.as_tensor(
                    self.get_token_embedding(token_id),
                    device=self.torch_device,
                    dtype=torch.float16,
                )

            for l_idx in range(self.config.n_layers):
                layer = self.get_layer(l_idx)
                x = layer.forward(x, pos, kv_cache, self.engine)

            if not compute_lm_head:
                return -1, 0.0

            x_norm = rmsnorm_torch(x, self.output_norm, eps=self.config.rms_norm_eps)

            if self.emb_cuda is not None:
                logits = F.linear(x_norm, self.emb_cuda)
                best_logit, best_id = torch.max(logits, dim=-1)
                if return_logits:
                    return int(best_id.item()), float(best_logit.item()), logits
                return int(best_id.item()), float(best_logit.item())

            best_id = (token_id + 1) % self.config.vocab_size
            best_logit = float(torch.dot(x_norm, x_norm).item())
            if return_logits:
                mock_logits = torch.zeros(
                    self.config.vocab_size,
                    device=self.torch_device,
                    dtype=torch.float32,
                )
                mock_logits[best_id] = best_logit
                return best_id, best_logit, mock_logits
            return best_id, best_logit

    def _forward_token_cpu(
        self,
        token_id: int,
        pos: int,
        kv_cache: KVCache,
        compute_lm_head: bool = True,
        return_logits: bool = False,
    ) -> tuple[int, float] | tuple[int, float, Any]:
        """Compute full forward pass for a single token on CPU P-cores."""
        # 1. Token Embedding Lookup
        x = self.get_token_embedding(token_id)

        # 2. Sequential Layers
        for l_idx in range(self.config.n_layers):
            layer = self.get_layer(l_idx)
            x = layer.forward(x, pos, kv_cache, self.engine)

        if not compute_lm_head:
            return -1, 0.0

        # 3. Output RMSNorm
        x = self.engine.rmsnorm(x, self.output_norm, eps=self.config.rms_norm_eps)

        # 4. LM Head argmax over full vocabulary
        if self.parser is None:
            next_token_id = (token_id + 1) % self.config.vocab_size
            logit = float(np.dot(x, x))
            if return_logits:
                mock_logits = np.zeros(self.config.vocab_size, dtype=np.float32)
                mock_logits[next_token_id] = logit
                return next_token_id, logit, mock_logits
            return next_token_id, logit

        if return_logits:
            logits = self.engine.lm_head_logits(
                self.emb_buffer, x, self.config.vocab_size
            )
            best_id = int(np.argmax(logits))
            best_logit = float(logits[best_id])
            return best_id, best_logit, logits

        next_token_id, logit = self.engine.lm_head_greedy(
            self.emb_buffer, x, self.config.vocab_size
        )
        return next_token_id, logit
