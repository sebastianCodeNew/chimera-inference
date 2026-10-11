"""Live verification script for Chimera-Inference OpenAI REST API."""

from pathlib import Path

from fastapi.testclient import TestClient

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
    print("=" * 60)
    print("Chimera-Inference: Live OpenAI API Verification")
    print("=" * 60)

    print("Initializing Generator on CUDA with 28 layers...")
    generator = ChimeraGenerator.from_gguf(DEFAULT_BLOB, n_layers=28, device="cuda")
    proposer = PromptLookupDraftProposer()
    coordinator = HeterogeneousSpeculativeCoordinator(
        target_generator=generator, draft_proposer=proposer, gamma=3
    )

    app = create_app(
        generator=generator,
        speculative_coordinator=coordinator,
        model_name="qwen2.5-1.5b-instruct",
    )
    client = TestClient(app)

    # 1. Health Probe
    res_health = client.get("/health")
    print("\n1. Health Probe Response:", res_health.json())
    assert res_health.status_code == 200

    # 2. Models Endpoint
    res_models = client.get("/v1/models")
    print("\n2. Models Endpoint Response:", res_models.json())
    assert res_models.status_code == 200

    # 3. Chat Completion Non-Streaming
    print("\n3. Testing POST /v1/chat/completions (Non-Streaming)...")
    res_chat = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-1.5b-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Jawab dalam 1 kalimat pendek: apa itu matahari?",
                }
            ],
            "max_tokens": 20,
            "temperature": 0.7,
            "min_p": 0.05,
        },
    )
    print("   Chat Completion Status:", res_chat.status_code)
    assert res_chat.status_code == 200
    choice = res_chat.json()["choices"][0]
    print("   Generated Content:", choice["message"]["content"])
    print("   Finish Reason:", choice["finish_reason"])
    print("   Usage:", res_chat.json().get("usage"))

    # 4. Chat Completion Streaming (SSE)
    print("\n4. Testing POST /v1/chat/completions (SSE Streaming)...")
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "qwen2.5-1.5b-instruct",
            "messages": [{"role": "user", "content": "Sebutkan angka 1 sampai 5."}],
            "max_tokens": 15,
            "stream": True,
        },
    ) as stream_res:
        print("   SSE Status:", stream_res.status_code)
        assert stream_res.status_code == 200
        streamed_chunks = []
        for line in stream_res.iter_lines():
            if line and line.startswith("data: ") and line != "data: [DONE]":
                import json

                payload = json.loads(line[6:])
                delta = payload["choices"][0]["delta"].get("content", "")
                if delta:
                    streamed_chunks.append(delta)
        print("   Streamed Text:", "".join(streamed_chunks))

    print("\n" + "=" * 60)
    print("All Live API Checks Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
