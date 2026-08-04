"""
HTTP API Server
===============

This module defines the FastAPI application, HTTP endpoints, router configuration,
and request/response validation schemas. It delegates core classification tasks
to the prediction module.
"""

import asyncio
import os
from typing import Any
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel, Field

from sulku.bootstrap import Settings, setup
from sulku.concurrency import ClassifySemaphore, EWMARate
from sulku.eviction import eviction_loop
from sulku.metrics import requests_shed, set_metrics_service
from sulku.models import ModelStore
from sulku.prediction import PredictionService
from sulku.wpapi import wpapi_router
from .constants import DEFAULT_ALPHA, DEFAULT_P_STAY
from sulku.utils import parse_paragraphs_and_sentences, strip_markdown

from niitti import get_logger

logger = get_logger(__name__)

_semaphore = ClassifySemaphore(max_concurrent=4, max_queue=32)
_rate = EWMARate(half_life=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with setup() as (settings, store):
        svc = PredictionService(store=store)
        svc.default_keep_alive = settings.keep_alive
        _semaphore.max_queue = settings.max_queue
        _semaphore._sem = asyncio.Semaphore(settings.max_concurrent)

        app.state.settings = settings
        app.state.model_store = store
        app.state.prediction_service = svc
        set_metrics_service(svc)

        preload = getattr(app.state, "preload", settings.preload)
        if preload:
            await asyncio.gather(*[svc.ensure_loaded(n) for n in store.model_names])
        task = asyncio.create_task(
            eviction_loop(svc), name="model-eviction"
        )
        try:
            yield
        finally:
            task.cancel()
            svc.clear_models()


def get_app_settings(request: Request) -> Settings:
    """FastAPI Dependency: get active Settings from app state."""
    return request.app.state.settings


def get_app_model_store(request: Request) -> ModelStore:
    """FastAPI Dependency: get active ModelStore from app state."""
    return request.app.state.model_store


def get_app_prediction_service(request: Request) -> Any:
    """FastAPI Dependency: get active PredictionService from app state."""
    return request.app.state.prediction_service


class ClassificationRequest(BaseModel):
    text: str = Field(
        ..., min_length=10, description="The document or sentence to analyze."
    )


class ParagraphDetailResponse(BaseModel):
    text: str = Field(..., description="The raw/trimmed text of the paragraph.")
    sentences: list[str] = Field(..., description="The list of sentences in the paragraph.")
    predictions: dict[str, float] | None = Field(
        None, description="Prediction score per model for this paragraph, or null if excluded."
    )
    final_score: float | None = Field(
        None, description="The ensemble average score for this paragraph, or null if excluded."
    )


class ClassificationResponse(BaseModel):
    is_ai: bool
    ai_votes: int
    total_models: int
    final_score: float
    final_confidence: float
    final_z_score: float = Field(0.0, description="Stouffer combined Z-score across models.")
    predictions: dict[str, float]
    confidences: dict[str, float]
    z_scores: dict[str, float] = Field(default_factory=dict, description="Per-model Z-scores.")
    paragraphs: list[ParagraphDetailResponse] = Field(
        ..., description="Per-paragraph predictions and metadata."
    )


class ModelInfo(BaseModel):
    name: str = Field(..., description="The identifier of the fasttext classifier model.")
    path: str = Field(..., description="Resolved absolute file path of the model.")
    loaded: bool = Field(..., description="Whether the model is currently loaded in memory.")


class ModelListResponse(BaseModel):
    models: list[ModelInfo] = Field(..., description="List of configured classifier models.")


class KeepAliveRequest(BaseModel):
    seconds: float = Field(
        ...,
        description="Idle TTL in seconds before a model is evicted. -1 disables TTL.",
    )


router = APIRouter(prefix="/api/v1/aidetect", tags=["classification"])


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    store: ModelStore = Depends(get_app_model_store),
    pred_svc: Any = Depends(get_app_prediction_service),
):
    """
    List all configured fastText classifier models and their load status.
    """
    result = []
    for name in store.model_names:
        is_loaded = name in pred_svc.models
        spec = store.get_spec(name)
        path_str = str(spec.source) if spec else ""
        result.append(
            ModelInfo(
                name=name,
                path=path_str,
                loaded=is_loaded,
            )
        )
    return ModelListResponse(models=result)


@router.post("/models/{name}/load")
async def load_model(
    name: str,
    pred_svc: Any = Depends(get_app_prediction_service),
):
    """
    Explicitly load a model into memory.
    """
    try:
        await pred_svc.ensure_loaded(name)
    except KeyError:
        raise HTTPException(404, f"Unknown model '{name}'.")
    return {"model": name, "loaded": True}


@router.post("/models/{name}/unload")
async def unload_model_endpoint(
    name: str,
    pred_svc: Any = Depends(get_app_prediction_service),
):
    """
    Explicitly unload a model from memory.
    """
    try:
        pred_svc.unload_model(name)
    except KeyError:
        raise HTTPException(404, f"Model '{name}' is not loaded.")
    return {"model": name, "loaded": False}


@router.patch("/models/keep-alive")
async def set_keep_alive(body: KeepAliveRequest, pred_svc: Any = Depends(get_app_prediction_service)):
    """
    Update default idle TTL across all models.
    """
    pred_svc.default_keep_alive = body.seconds
    return {"keep_alive": pred_svc.default_keep_alive}


def verify_plain_text(body_bytes: bytes) -> None:
    """
    Verify that the request body bytes represent plain text and not binary content.
    Raises HTTPException 415 if it is binary.
    """
    # Check for null bytes (common indicator of binary files)
    if b"\x00" in body_bytes:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type: Binary content detected (contains null bytes).",
        )

    # Verify it can be decoded as UTF-8
    try:
        body_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Content-Type: Binary or invalid UTF-8 content detected.",
        )



@router.post("/", response_model=ClassificationResponse)
async def classify_text(
    req: Request,
    p_stay: float = DEFAULT_P_STAY,
    alpha: float = DEFAULT_ALPHA,
    models: list[str] | None = Query(
        None,
        description="Optional list of specific model names to evaluate. Omit to evaluate all loaded models.",
    ),
    store: ModelStore = Depends(get_app_model_store),
    pred_svc: Any = Depends(get_app_prediction_service),
):
    # Burst onset: EWMA ticks non-zero -> pre-load all models speculatively
    if _rate.tick():
        for name in store.model_names:
            asyncio.create_task(pred_svc.ensure_loaded(name))

    target_names = models if models else store.model_names
    for name in target_names:
        try:
            await pred_svc.ensure_loaded(name)
        except KeyError:
            raise HTTPException(422, f"Unknown model '{name}'.")

    content_type = req.headers.get("content-type", "")
    main_type = (
        content_type.split(";")[0].strip().lower() if content_type else "text/plain"
    )

    # Read raw body bytes to verify content
    body_bytes = await req.body()
    verify_plain_text(body_bytes)

    is_markdown_content = False

    # Extract text content and detect if it is markdown based on Content-Type using match
    match main_type:
        case "text/markdown" | "text/x-markdown":
            is_markdown_content = True
            text = body_bytes.decode("utf-8")
        case "text/plain" | "":
            text = body_bytes.decode("utf-8")
        case _:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported Content-Type '{content_type}': "
                       f"Only text/plain, text/markdown, and application/json are allowed.",
            )

    # Apply minimum length validation on raw input
    if len(text) < 10:
        raise HTTPException(
            status_code=422,
            detail="Text content is too short (less than 10 characters).",
        )

    # If it is markdown, strip formatting and frontmatter
    if is_markdown_content:
        text = strip_markdown(text).strip()
        if len(text) < 10:
            raise HTTPException(
                status_code=422,
                detail="Text content is too short (less than 10 characters) "
                       "after stripping markdown and frontmatter.",
            )

    # Parse text into paragraphs and sentences first
    try:
        parsed_paragraphs = parse_paragraphs_and_sentences(text)
    except Exception as exc:
        logger.exception("Failed to parse text: %s", exc)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse text: {exc}",
        )

    if not parsed_paragraphs:
        raise HTTPException(
            status_code=422,
            detail="Text content does not contain classifiable paragraphs.",
        )

    try:
        async with _semaphore:
            with logger.span("classify_text_endpoint", p_stay=p_stay, alpha=alpha):
                res = pred_svc.classify(parsed_paragraphs, p_stay=p_stay, alpha=alpha, models=models)
    except ClassifySemaphore.OverloadError as exc:
        requests_shed.add(1)
        raise HTTPException(
            status_code=503,
            headers={"Retry-After": "5"},
            detail=str(exc),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ClassificationResponse(
        is_ai=res.is_ai,
        ai_votes=res.ai_votes,
        total_models=res.total_models,
        final_score=res.final_score,
        final_confidence=res.final_confidence,
        final_z_score=res.final_z_score,
        predictions=res.predictions,
        confidences=res.confidences,
        z_scores=res.z_scores,
        paragraphs=[
            ParagraphDetailResponse(
                text=p.text,
                sentences=p.sentences,
                predictions=p.predictions,
                final_score=p.final_score,
            )
            for p in res.paragraphs
        ],
    )


def create_app(preload: bool = False) -> FastAPI:
    from prometheus_client import make_asgi_app
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    app = FastAPI(title="AI Text Classifier Service", lifespan=lifespan)
    app.state.preload = preload
    FastAPIInstrumentor.instrument_app(app)
    app.mount("/metrics", make_asgi_app())

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(wpapi_router)
    app.include_router(router)

    return app


def create_app_from_env() -> FastAPI:
    """
    Factory function for Uvicorn when starting with ``--reload`` (``factory=True``).

    When Uvicorn runs in reload mode, it spawns separate child worker processes that
    re-import application modules upon file changes. Passing the import path string
    ``"sulku.http:create_app_from_env"`` allows child workers to independently instantiate
    the app while reading environment settings (such as ``SULKU_PRELOAD``).
    """
    preload = os.getenv("SULKU_PRELOAD", "false").lower() in ("true", "1")
    return create_app(preload=preload)


