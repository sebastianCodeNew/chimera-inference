"""Production-Grade OpenAI-Compatible REST API Server for Chimera-Inference Engine.

Exposes high-performance OpenAI API endpoints:
- POST /v1/chat/completions (Non-streaming and SSE Streaming)
- GET /v1/models (Model cards, devices, active architecture)
- GET /health (Hardware status, GPU VRAM metrics, P-core topology)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

from src.generator import ChimeraGenerator
from src.sampling import SamplingParams
from src.speculative import HeterogeneousSpeculativeCoordinator

# ============================================================================
# OpenAI API Schemas (Pydantic v2)
# ============================================================================


class ChatMessage(BaseModel):
    """Chat message object."""

    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request schema."""

    model: str = "chimera-default"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None
    use_speculative: bool = False


class ChatCompletionResponseChoice(BaseModel):
    """Individual choice in chat completion response."""

    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage tracking."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Standard OpenAI chat completion response schema."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    """Incremental chunk content for Server-Sent Events."""

    role: str | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    """Choice representation in streaming chunk."""

    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI SSE stream chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


class ModelCard(BaseModel):
    """Representation of an available model."""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "chimera"
    device: str
    active_layers: int
    max_seq_len: int


class ModelListResponse(BaseModel):
    """List of models schema."""

    object: str = "list"
    data: list[ModelCard]


class HealthResponse(BaseModel):
    """Hardware health, readiness, and GPU VRAM probe."""

    status: str = "ok"
    device: str
    active_model: str
    vram_allocated_mb: float | None = None
    vram_reserved_mb: float | None = None
    p_cores_count: int
    total_cores_count: int


class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request schema."""

    model: str = "chimera-default"
    prompt: str | list[str]
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=0.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    max_tokens: int | None = Field(default=16, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None
    use_speculative: bool = False


class CompletionResponseChoice(BaseModel):
    """Choice representation in text completion response."""

    index: int = 0
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """Standard OpenAI text completion response schema."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


# ============================================================================
# FastAPI Application Factory
# ============================================================================


def create_app(
    generator: ChimeraGenerator | None = None,
    speculative_coordinator: HeterogeneousSpeculativeCoordinator | None = None,
    model_name: str = "chimera-default",
) -> FastAPI:
    """Create and configure production FastAPI application."""
    app = FastAPI(
        title="Chimera-Inference Engine API",
        version="1.0.0",
        description="High-performance OpenAI-compatible REST server for Chimera-Inference.",
    )

    # Allow CORS for web clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store state on app
    app.state.generator = generator
    app.state.speculative_coordinator = speculative_coordinator
    app.state.model_name = model_name

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": exc.detail,
                    "type": "invalid_request_error"
                    if exc.status_code < 500
                    else "api_error",
                    "code": exc.status_code,
                }
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "message": str(exc),
                    "type": "api_error",
                    "code": 500,
                }
            },
        )

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health and readiness probe with VRAM and CPU topology inspection."""
        gen: ChimeraGenerator | None = app.state.generator
        if gen is None:
            return HealthResponse(
                status="initializing",
                device="unknown",
                active_model=app.state.model_name,
                p_cores_count=0,
                total_cores_count=0,
            )

        device = gen.model.device
        vram_alloc: float | None = None
        vram_res: float | None = None

        if (
            device == "cuda"
            and HAS_TORCH
            and torch is not None
            and torch.cuda.is_available()
        ):
            vram_alloc = round(torch.cuda.memory_allocated() / (1024**2), 2)
            vram_res = round(torch.cuda.memory_reserved() / (1024**2), 2)

        p_cores = getattr(gen.model.engine, "p_cores_count", 0)
        total_cores = getattr(gen.model.engine, "total_cores_count", 0)

        return HealthResponse(
            status="ok",
            device=device,
            active_model=app.state.model_name,
            vram_allocated_mb=vram_alloc,
            vram_reserved_mb=vram_res,
            p_cores_count=p_cores,
            total_cores_count=total_cores,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models() -> ModelListResponse:
        """List active models and device capabilities."""
        gen: ChimeraGenerator | None = app.state.generator
        created_ts = int(time.time())

        if gen is None:
            return ModelListResponse(data=[])

        card = ModelCard(
            id=app.state.model_name,
            created=created_ts,
            device=gen.model.device,
            active_layers=gen.model.config.n_layers,
            max_seq_len=gen.max_seq_len,
        )
        return ModelListResponse(data=[card])

    @app.get("/v1/models/{model_id}", response_model=ModelCard)
    async def get_model(model_id: str) -> ModelCard:
        """Retrieve details of a specific active model."""
        gen: ChimeraGenerator | None = app.state.generator
        if gen is None or model_id != app.state.model_name:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found",
            )
        return ModelCard(
            id=app.state.model_name,
            created=int(time.time()),
            device=gen.model.device,
            active_layers=gen.model.config.n_layers,
            max_seq_len=gen.max_seq_len,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        """Generate chat completions conforming to OpenAI API specifications."""
        gen: ChimeraGenerator | None = app.state.generator
        if gen is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine is not initialized",
            )

        if not req.messages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="messages cannot be empty",
            )

        max_new_tokens = req.max_tokens or req.max_completion_tokens or 64

        sampling_params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            min_p=req.min_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            seed=req.seed,
        )

        raw_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        # Handle SSE Streaming
        if req.stream:

            async def event_generator() -> AsyncIterator[str]:
                # 1. Send initial role chunk
                initial_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=req.model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=DeltaMessage(role="assistant"),
                            finish_reason=None,
                        )
                    ],
                )
                yield f"data: {initial_chunk.model_dump_json()}\n\n"

                # 2. Stream generation chunks (support speculative streaming if enabled)
                if (
                    req.use_speculative
                    and app.state.speculative_coordinator is not None
                ):
                    stream_iter = app.state.speculative_coordinator.chat_stream(
                        raw_messages,
                        max_new_tokens=max_new_tokens,
                        stop=req.stop,
                        sampling_params=sampling_params,
                    )
                else:
                    stream_iter = gen.chat_stream(
                        raw_messages,
                        max_new_tokens=max_new_tokens,
                        stop=req.stop,
                        sampling_params=sampling_params,
                    )

                for token_str in stream_iter:
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created_ts,
                        model=req.model,
                        choices=[
                            ChunkChoice(
                                index=0,
                                delta=DeltaMessage(content=token_str),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"

                # 3. Final chunk with authentic finish_reason ("stop" or "length")
                finish_reason = getattr(gen, "last_finish_reason", "stop")
                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created_ts,
                    model=req.model,
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json()}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming execution
        if req.use_speculative and app.state.speculative_coordinator is not None:
            coord: HeterogeneousSpeculativeCoordinator = (
                app.state.speculative_coordinator
            )
            response_text = coord.chat(
                raw_messages,
                max_new_tokens=max_new_tokens,
                stop=req.stop,
                sampling_params=sampling_params,
            )
        else:
            response_text = gen.chat(
                raw_messages,
                max_new_tokens=max_new_tokens,
                stop=req.stop,
                sampling_params=sampling_params,
            )

        finish_reason = getattr(gen, "last_finish_reason", "stop")

        # Estimate token usage
        prompt_str = gen.tokenizer.apply_chat_template(
            raw_messages, add_generation_prompt=True
        )
        prompt_tokens_len = len(gen.tokenizer.encode(prompt_str))
        completion_tokens_len = len(gen.tokenizer.encode(response_text))

        response = ChatCompletionResponse(
            id=completion_id,
            created=created_ts,
            model=req.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_tokens_len + completion_tokens_len,
            ),
        )
        return JSONResponse(content=response.model_dump())

    @app.post("/v1/completions")
    async def text_completions(req: CompletionRequest) -> Response:
        """Generate legacy text completions conforming to OpenAI API specifications."""
        gen: ChimeraGenerator | None = app.state.generator
        if gen is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine is not initialized",
            )

        prompt_str = (
            req.prompt if isinstance(req.prompt, str) else "\n".join(req.prompt)
        )
        if not prompt_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="prompt cannot be empty",
            )

        max_new_tokens = req.max_tokens or 16
        sampling_params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            min_p=req.min_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            frequency_penalty=req.frequency_penalty,
            presence_penalty=req.presence_penalty,
            seed=req.seed,
        )

        completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
        created_ts = int(time.time())

        if req.stream:

            async def event_generator() -> AsyncIterator[str]:
                if (
                    req.use_speculative
                    and app.state.speculative_coordinator is not None
                ):
                    stream_iter = app.state.speculative_coordinator.generate_stream(
                        prompt_str,
                        max_new_tokens=max_new_tokens,
                        stop=req.stop,
                        sampling_params=sampling_params,
                    )
                else:
                    stream_iter = gen.generate_stream(
                        prompt_str,
                        max_new_tokens=max_new_tokens,
                        stop=req.stop,
                        sampling_params=sampling_params,
                    )

                for token_str in stream_iter:
                    chunk = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created_ts,
                        "model": req.model,
                        "choices": [
                            {"index": 0, "text": token_str, "finish_reason": None}
                        ],
                    }
                    import json

                    yield f"data: {json.dumps(chunk)}\n\n"

                finish_reason = getattr(gen, "last_finish_reason", "stop")
                final_chunk = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created_ts,
                    "model": req.model,
                    "choices": [
                        {"index": 0, "text": "", "finish_reason": finish_reason}
                    ],
                }
                import json

                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        if req.use_speculative and app.state.speculative_coordinator is not None:
            coord: HeterogeneousSpeculativeCoordinator = (
                app.state.speculative_coordinator
            )
            response_text = coord.generate(
                prompt_str,
                max_new_tokens=max_new_tokens,
                stop=req.stop,
                sampling_params=sampling_params,
            )
        else:
            response_text = gen.generate(
                prompt_str,
                max_new_tokens=max_new_tokens,
                stop=req.stop,
                sampling_params=sampling_params,
            )

        finish_reason = getattr(gen, "last_finish_reason", "stop")
        prompt_tokens_len = len(gen.tokenizer.encode(prompt_str))
        completion_tokens_len = len(gen.tokenizer.encode(response_text))

        response = CompletionResponse(
            id=completion_id,
            created=created_ts,
            model=req.model,
            choices=[
                CompletionResponseChoice(
                    index=0,
                    text=response_text,
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens_len,
                completion_tokens=completion_tokens_len,
                total_tokens=prompt_tokens_len + completion_tokens_len,
            ),
        )
        return JSONResponse(content=response.model_dump())

    return app
