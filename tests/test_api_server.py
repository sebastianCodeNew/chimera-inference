"""Unit tests for OpenAI-Compatible REST API Server."""

import pytest
from starlette.testclient import TestClient

from src.api_server import create_app
from src.generator import ChimeraGenerator
from src.model import ChimeraModel
from src.speculative import HeterogeneousSpeculativeCoordinator
from src.tokenizer import ChimeraTokenizer


@pytest.fixture
def mock_generator() -> ChimeraGenerator:
    model = ChimeraModel.create_mock()
    tokenizer = ChimeraTokenizer.create_mock()
    return ChimeraGenerator(model, tokenizer, max_seq_len=64)


@pytest.fixture
def client(mock_generator: ChimeraGenerator) -> TestClient:
    coordinator = HeterogeneousSpeculativeCoordinator(mock_generator, gamma=2)
    app = create_app(
        generator=mock_generator,
        speculative_coordinator=coordinator,
        model_name="mock-chimera",
    )
    return TestClient(app)


def test_health_endpoint_uninitialized():
    app = create_app(generator=None, model_name="uninit")
    client = TestClient(app)
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "initializing"
    assert data["device"] == "unknown"


def test_health_endpoint_initialized(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["device"] == "cpu"
    assert data["active_model"] == "mock-chimera"
    assert "p_cores_count" in data


def test_models_endpoint(client: TestClient):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    model = data["data"][0]
    assert model["id"] == "mock-chimera"
    assert model["owned_by"] == "chimera"
    assert model["device"] == "cpu"
    assert model["active_layers"] == 2
    assert model["max_seq_len"] == 64


def test_chat_completions_non_streaming(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 4,
        "temperature": 0.0,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-chimera"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ["stop", "length"]
    assert "usage" in data
    assert data["usage"]["completion_tokens"] > 0


def test_chat_completions_with_sampling_params(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 4,
        "temperature": 0.8,
        "top_p": 0.9,
        "min_p": 0.05,
        "top_k": 20,
        "repetition_penalty": 1.1,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["choices"]) == 1
    assert isinstance(data["choices"][0]["message"]["content"], str)


def test_chat_completions_speculative(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 5,
        "use_speculative": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["choices"]) == 1
    assert isinstance(data["choices"][0]["message"]["content"], str)


def test_chat_completions_streaming(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 3,
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]

    lines = res.text.strip().split("\n\n")
    assert len(lines) >= 3

    # Check first line has role chunk
    first_chunk = lines[0]
    assert first_chunk.startswith("data: ")

    # Check last line is [DONE]
    assert lines[-1] == "data: [DONE]"


def test_chat_completions_empty_messages(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [],
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 400
    data = res.json()
    assert "error" in data
    assert data["error"]["type"] == "invalid_request_error"


def test_chat_completions_uninitialized_engine():
    app = create_app(generator=None)
    client = TestClient(app)
    payload = {
        "messages": [{"role": "user", "content": "Halo"}],
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 503
    data = res.json()
    assert "error" in data


def test_get_model_by_id(client: TestClient):
    # Found
    res = client.get("/v1/models/mock-chimera")
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "mock-chimera"
    assert data["device"] == "cpu"

    # Not found
    res404 = client.get("/v1/models/non-existent-model")
    assert res404.status_code == 404
    assert res404.json()["error"]["code"] == 404


def test_chat_completions_speculative_streaming(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 4,
        "stream": True,
        "use_speculative": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    lines = res.text.strip().split("\n\n")
    assert len(lines) >= 3
    assert lines[-1] == "data: [DONE]"


def test_chat_completions_stop_sequence(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 10,
        "stop": "\n",
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert len(data["choices"]) == 1


def test_text_completions_non_streaming(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "prompt": "Halo",
        "max_tokens": 4,
    }
    res = client.post("/v1/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) == 1
    assert isinstance(data["choices"][0]["text"], str)
    assert "usage" in data


def test_text_completions_streaming(client: TestClient):
    payload = {
        "model": "mock-chimera",
        "prompt": "Halo",
        "max_tokens": 4,
        "stream": True,
    }
    res = client.post("/v1/completions", json=payload)
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    lines = res.text.strip().split("\n\n")
    assert len(lines) >= 2
    assert lines[-1] == "data: [DONE]"
