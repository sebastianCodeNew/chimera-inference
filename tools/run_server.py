"""Production CLI Launcher for Chimera-Inference OpenAI-Compatible REST Server."""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

import uvicorn

from src.api_server import create_app
from src.generator import ChimeraGenerator
from src.speculative import (
    HeterogeneousSpeculativeCoordinator,
    PromptLookupDraftProposer,
)

DEFAULT_BLOB = Path(
    "/home/sebastian/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chimera-Inference Engine: Production REST Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Bind network interface host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Target acceleration hardware: 'cpu', 'cuda', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=28,
        help="Number of transformer layers to load (default: 28)",
    )
    parser.add_argument(
        "--blob",
        type=Path,
        default=DEFAULT_BLOB,
        help=f"Path to GGUF model file (default: {DEFAULT_BLOB})",
    )
    parser.add_argument(
        "--speculative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable heterogeneous speculative decoding coordinator (default: True, use --no-speculative to disable)",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=3,
        help="Speculative draft depth gamma (default: 3)",
    )

    args = parser.parse_args()

    target_device = args.device
    if target_device == "auto":
        target_device = (
            "cuda"
            if (HAS_TORCH and torch is not None and torch.cuda.is_available())
            else "cpu"
        )

    device_label = (
        f"CUDA ({torch.cuda.get_device_name(0)})"
        if (target_device == "cuda" and HAS_TORCH and torch is not None)
        else "CPU (Intel P-cores AVX2)"
    )

    print("=" * 70)
    print("Chimera-Inference Engine: Production OpenAI-Compatible REST API")
    print(f"Model Blob: {args.blob}")
    print(f"Hardware Acceleration: {device_label}")
    print(f"Active Layers: {args.layers} | Speculative Decoding: {args.speculative}")
    print(f"Listening on: http://{args.host}:{args.port}")
    print("=" * 70)

    if not args.blob.exists():
        print(f"ERROR: Model blob {args.blob} not found!", file=sys.stderr)
        sys.exit(1)

    print("[1/2] Loading Chimera-Inference Generator...")
    generator = ChimeraGenerator.from_gguf(
        args.blob,
        n_layers=args.layers,
        max_cached_layers=args.layers,
        preload_layers=True,
        device=target_device,
    )

    coordinator = None
    if args.speculative:
        print("[2/2] Initializing Heterogeneous Speculative Decoding Coordinator...")
        proposer = PromptLookupDraftProposer()
        coordinator = HeterogeneousSpeculativeCoordinator(
            target_generator=generator,
            draft_proposer=proposer,
            gamma=args.gamma,
        )

    if target_device == "cuda" and HAS_TORCH and torch is not None:
        vram_alloc = torch.cuda.memory_allocated() / (1024**2)
        vram_res = torch.cuda.memory_reserved() / (1024**2)
        print(
            f"GPU VRAM Allocated: {vram_alloc:.1f} MiB | Reserved: {vram_res:.1f} MiB"
        )

    print("\nStarting Uvicorn Server...")
    app = create_app(
        generator=generator,
        speculative_coordinator=coordinator,
        model_name="qwen2.5-0.5b-instruct",
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
