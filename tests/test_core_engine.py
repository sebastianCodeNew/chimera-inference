"""Unit tests for ChimeraCoreEngine."""

import numpy as np
import pytest

from src.core_engine import ChimeraCoreEngine


@pytest.fixture
def engine() -> ChimeraCoreEngine:
    return ChimeraCoreEngine()


def test_core_engine_topology(engine: ChimeraCoreEngine):
    assert engine.p_cores_count >= 1
    assert engine.total_cores_count >= engine.p_cores_count


@pytest.mark.parametrize(
    "rows,cols", [(64, 128), (512, 1024), (2048, 1536), (4096, 4096)]
)
def test_gemv_correctness_against_numpy(
    engine: ChimeraCoreEngine, rows: int, cols: int
):
    np.random.seed(42)
    W = np.random.randn(rows, cols).astype(np.float32)
    x = np.random.randn(cols).astype(np.float32)

    expected = np.dot(W, x)

    # Test P-cores pinned GEMV
    actual_p_cores = engine.gemv_p_cores(W, x)
    assert np.allclose(actual_p_cores, expected, atol=1e-4, rtol=1e-4)

    # Test Unpinned GEMV
    actual_unpinned = engine.gemv_unpinned(W, x)
    assert np.allclose(actual_unpinned, expected, atol=1e-4, rtol=1e-4)


def test_gemv_dimension_mismatch(engine: ChimeraCoreEngine):
    W = np.zeros((128, 64), dtype=np.float32)
    x = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        engine.gemv_p_cores(W, x)


def test_gemv_dtype_validation(engine: ChimeraCoreEngine):
    W = np.zeros((64, 64), dtype=np.float64)
    x = np.zeros(64, dtype=np.float32)
    with pytest.raises(TypeError, match="float32 dtype"):
        engine.gemv_p_cores(W, x)  # type: ignore[arg-type]


def test_rmsnorm_correctness(engine: ChimeraCoreEngine):
    np.random.seed(42)
    x = np.random.randn(1536).astype(np.float32)
    w = np.random.randn(1536).astype(np.float32)
    eps = 1e-6

    expected = (x / np.sqrt(np.mean(x**2) + eps)) * w
    actual = engine.rmsnorm(x, w, eps=eps)

    assert np.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_rmsnorm_validation(engine: ChimeraCoreEngine):
    x = np.zeros(128, dtype=np.float32)
    w = np.zeros(64, dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        engine.rmsnorm(x, w)

    with pytest.raises(TypeError, match="float32 dtype"):
        engine.rmsnorm(x.astype(np.float64), w.astype(np.float64))  # type: ignore[arg-type]


def test_gemv_q4_k_correctness_against_dequantized(engine: ChimeraCoreEngine):
    import struct

    np.random.seed(42)
    rows, cols = 64, 256
    bytes_per_row = (cols // 256) * 144
    total_bytes = rows * bytes_per_row
    raw_bytes = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)

    # Set valid non-NaN fp16 scales (d=1.0, dmin=0.5) for each block
    valid_header = np.frombuffer(struct.pack("<HH", 0x3C00, 0x3800), dtype=np.uint8)
    for b in range(rows * (cols // 256)):
        offset = b * 144
        raw_bytes[offset : offset + 4] = valid_header

    x = np.random.randn(cols).astype(np.float32)

    # Actual output from fast fused quant GEMV
    actual = engine.gemv_q4_k_p_cores(raw_bytes, x, rows, cols)

    # Expected from dequantizing matrix and dot product
    expected = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        row_raw = raw_bytes[r * bytes_per_row : (r + 1) * bytes_per_row]
        row_f32 = engine.dequantize_q4_k(row_raw, cols)
        expected[r] = np.dot(row_f32, x)

    assert np.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_gemv_q6_k_correctness_against_dequantized(engine: ChimeraCoreEngine):
    import struct

    np.random.seed(42)
    rows, cols = 64, 256
    bytes_per_row = (cols // 256) * 210
    total_bytes = rows * bytes_per_row
    raw_bytes = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)

    # Set valid non-NaN fp16 scale (d=1.0) for each block
    valid_d = np.frombuffer(struct.pack("<H", 0x3C00), dtype=np.uint8)
    for b in range(rows * (cols // 256)):
        offset = b * 210
        raw_bytes[offset + 208 : offset + 210] = valid_d

    x = np.random.randn(cols).astype(np.float32)

    # Actual output from fast fused quant GEMV
    actual = engine.gemv_q6_k_p_cores(raw_bytes, x, rows, cols)

    # Expected from dequantizing matrix and dot product
    expected = np.empty(rows, dtype=np.float32)
    for r in range(rows):
        row_raw = raw_bytes[r * bytes_per_row : (r + 1) * bytes_per_row]
        row_f32 = engine.dequantize_q6_k(row_raw, cols)
        expected[r] = np.dot(row_f32, x)

    assert np.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_quant_gemv_validations(engine: ChimeraCoreEngine):
    raw_bytes = np.zeros(144, dtype=np.uint8)
    x = np.zeros(256, dtype=np.float32)
    x_128 = np.zeros(128, dtype=np.float32)

    # Invalid cols not multiple of 256
    with pytest.raises(ValueError, match="multiple of 256"):
        engine.gemv_q4_k_p_cores(raw_bytes, x_128, 1, 128)

    # Vector length mismatch
    with pytest.raises(ValueError, match="dimension"):
        engine.gemv_q4_k_p_cores(raw_bytes, np.zeros(512, dtype=np.float32), 1, 256)

    # Buffer too small
    with pytest.raises(ValueError, match="buffer size"):
        engine.gemv_q4_k_p_cores(np.zeros(100, dtype=np.uint8), x, 1, 256)


def test_lm_head_logits_matches_greedy(engine: ChimeraCoreEngine):
    import struct

    np.random.seed(42)
    vocab_size = 128
    hidden_dim = 256
    bytes_per_token = (hidden_dim // 256) * 210
    total_bytes = vocab_size * bytes_per_token
    raw_bytes = np.random.randint(0, 256, size=total_bytes, dtype=np.uint8)

    # Set valid fp16 scale
    valid_d = np.frombuffer(struct.pack("<H", 0x3C00), dtype=np.uint8)
    for v in range(vocab_size):
        offset = v * bytes_per_token
        raw_bytes[offset + 208 : offset + 210] = valid_d

    x = np.random.randn(hidden_dim).astype(np.float32)

    best_id, best_logit = engine.lm_head_greedy(raw_bytes, x, vocab_size)
    logits = engine.lm_head_logits(raw_bytes, x, vocab_size)

    assert logits.shape == (vocab_size,)
    assert int(np.argmax(logits)) == best_id
    assert np.isclose(logits[best_id], best_logit, atol=1e-5)
