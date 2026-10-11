"""GGUF Binary Parser & Metadata Extractor for Chimera-Inference Engine.

Provides zero-copy memory-mapped inspection of GGUF model files, reading
architectural hyperparameters, tensor directories, and quantization types.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, BinaryIO, Self

import numpy as np


class GGUFValueType(IntEnum):
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    n_dims: int
    shape: tuple[int, ...]
    ggml_type: GGMLType
    offset: int
    size_bytes: int


@dataclass(frozen=True)
class GGUFHeader:
    magic: bytes
    version: int
    tensor_count: int
    metadata_kv_count: int


class GGUFParser:
    """Zero-copy streaming and memory-mapped reader for GGUF model files."""

    MAGIC: bytes = b"GGUF"

    def __init__(self, file_path: Path | str, load_vocab: bool = False) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"GGUF file not found: {self.file_path}")

        self.load_vocab = load_vocab
        self.metadata: dict[str, Any] = {}
        self.tensors: dict[str, GGUFTensorInfo] = {}
        self.header: GGUFHeader | None = None
        self.tensor_data_offset: int = 0
        self._mmap: mmap.mmap | None = None
        self._file_handle: BinaryIO | None = None

        self._parse()

    def _read_string(self, f: BinaryIO) -> str:
        length_bytes = f.read(8)
        if len(length_bytes) < 8:
            raise ValueError("Unexpected EOF while reading string length")
        length = struct.unpack("<Q", length_bytes)[0]
        string_bytes = f.read(length)
        if len(string_bytes) < length:
            raise ValueError("Unexpected EOF while reading string content")
        return string_bytes.decode("utf-8", errors="replace")

    def _read_value(self, f: BinaryIO, val_type: GGUFValueType) -> Any:
        match val_type:
            case GGUFValueType.UINT8:
                return struct.unpack("<B", f.read(1))[0]
            case GGUFValueType.INT8:
                return struct.unpack("<b", f.read(1))[0]
            case GGUFValueType.UINT16:
                return struct.unpack("<H", f.read(2))[0]
            case GGUFValueType.INT16:
                return struct.unpack("<h", f.read(2))[0]
            case GGUFValueType.UINT32:
                return struct.unpack("<I", f.read(4))[0]
            case GGUFValueType.INT32:
                return struct.unpack("<i", f.read(4))[0]
            case GGUFValueType.FLOAT32:
                return struct.unpack("<f", f.read(4))[0]
            case GGUFValueType.BOOL:
                return struct.unpack("<?", f.read(1))[0]
            case GGUFValueType.STRING:
                return self._read_string(f)
            case GGUFValueType.ARRAY:
                raw_elem_type = struct.unpack("<I", f.read(4))[0]
                elem_type = GGUFValueType(raw_elem_type)
                n_elements = struct.unpack("<Q", f.read(8))[0]
                # Avoid loading massive vocabulary arrays entirely in memory unless requested
                if (
                    not self.load_vocab
                    and n_elements > 1000
                    and elem_type == GGUFValueType.STRING
                ):
                    # Skip tokens array efficiently
                    for _ in range(n_elements):
                        str_len = struct.unpack("<Q", f.read(8))[0]
                        f.seek(str_len, 1)
                    return f"[Array of {n_elements} strings - omitted for performance]"
                return [self._read_value(f, elem_type) for _ in range(n_elements)]
            case GGUFValueType.UINT64:
                return struct.unpack("<Q", f.read(8))[0]
            case GGUFValueType.INT64:
                return struct.unpack("<q", f.read(8))[0]
            case GGUFValueType.FLOAT64:
                return struct.unpack("<d", f.read(8))[0]
            case _:
                raise ValueError(f"Unknown GGUF value type: {val_type}")

    def _parse(self) -> None:
        with open(self.file_path, "rb") as f:
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError(
                    f"Invalid GGUF magic: expected {self.MAGIC}, got {magic}"
                )

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            self.header = GGUFHeader(
                magic=magic,
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
            )

            # Parse Metadata
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                val_type = GGUFValueType(struct.unpack("<I", f.read(4))[0])
                val = self._read_value(f, val_type)
                self.metadata[key] = val

            # Parse Tensor Directory
            for _ in range(tensor_count):
                tensor_name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
                ggml_type = GGMLType(struct.unpack("<I", f.read(4))[0])
                offset = struct.unpack("<Q", f.read(8))[0]

                # Estimated size in bytes based on type and element count
                n_elements = 1
                for dim in shape:
                    n_elements *= dim

                self.tensors[tensor_name] = GGUFTensorInfo(
                    name=tensor_name,
                    n_dims=n_dims,
                    shape=shape,
                    ggml_type=ggml_type,
                    offset=offset,
                    size_bytes=n_elements,  # Placeholder, exact depends on block quant
                )

            current_pos = f.tell()
            alignment = self.metadata.get("general.alignment", 32)
            self.tensor_data_offset = (
                (current_pos + alignment - 1) // alignment
            ) * alignment

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    @property
    def block_count(self) -> int:
        arch = self.architecture
        return int(self.metadata.get(f"{arch}.block_count", 0))

    @property
    def embedding_length(self) -> int:
        arch = self.architecture
        return int(self.metadata.get(f"{arch}.embedding_length", 0))

    @property
    def head_count(self) -> int:
        arch = self.architecture
        return int(self.metadata.get(f"{arch}.attention.head_count", 0))

    @property
    def head_count_kv(self) -> int:
        arch = self.architecture
        return int(self.metadata.get(f"{arch}.attention.head_count_kv", 0))

    def load_tensor(
        self, name: str, engine: Any | None = None, dequantize: bool = True
    ) -> np.ndarray[Any, Any]:
        """Load a tensor by name into a NumPy array (dequantized or raw quantized)."""
        if name not in self.tensors:
            raise KeyError(f"Tensor '{name}' not found in model")

        info = self.tensors[name]
        target_shape = info.shape[::-1]
        n_elements = 1
        for d in info.shape:
            n_elements *= d

        with open(self.file_path, "rb") as f:
            f.seek(self.tensor_data_offset + info.offset)
            match info.ggml_type:
                case GGMLType.F32:
                    raw = f.read(n_elements * 4)
                    return (
                        np.frombuffer(raw, dtype=np.float32)
                        .copy()
                        .reshape(target_shape)
                    )
                case GGMLType.Q4_K:
                    n_blocks = n_elements // 256
                    raw = f.read(n_blocks * 144)
                    if not dequantize:
                        return np.frombuffer(raw, dtype=np.uint8).copy()
                    if engine is None:
                        from src.core_engine import ChimeraCoreEngine

                        engine = ChimeraCoreEngine()
                    return engine.dequantize_q4_k(raw, n_elements).reshape(target_shape)
                case GGMLType.Q6_K:
                    n_blocks = n_elements // 256
                    raw = f.read(n_blocks * 210)
                    if not dequantize:
                        return np.frombuffer(raw, dtype=np.uint8).copy()
                    if engine is None:
                        from src.core_engine import ChimeraCoreEngine

                        engine = ChimeraCoreEngine()
                    return engine.dequantize_q6_k(raw, n_elements).reshape(target_shape)
                case _:
                    raise NotImplementedError(
                        f"Unsupported GGML tensor type: {info.ggml_type.name}"
                    )

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
