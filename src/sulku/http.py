"""
HTTP API Server
===============

This module defines the FastAPI application, HTTP endpoints, router configuration,
and request/response validation schemas. It delegates core classification tasks
to the prediction module.
"""

import logging
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel, Field

from sulku.wpapi import wpapi_router
from .constants import DEFAULT_ALPHA, DEFAULT_P_STAY
from sulku.prediction import prediction_service
from sulku.utils import parse_paragraphs_and_sentences, strip_markdown

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE on server startup
    try:
        prediction_service.load_models()
        yield
    finally:
        # Clean up resources on shutdown if necessary
        prediction_service.clear_models()


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
    predictions: dict[str, float]
    confidences: dict[str, float]
    paragraphs: list[ParagraphDetailResponse] = Field(
        ..., description="Per-paragraph predictions and metadata."
    )



router = APIRouter(prefix="/api/v1/aidetect", tags=["classification"])


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
):
    if not prediction_service.is_initialized:
        raise HTTPException(status_code=500, detail="Models not initialized.")

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
        res = prediction_service.classify(parsed_paragraphs, p_stay=p_stay, alpha=alpha)
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
        predictions=res.predictions,
        confidences=res.confidences,
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



def create_app() -> FastAPI:
    app = FastAPI(title="AI Text Classifier Service", lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(wpapi_router)
    app.include_router(router)

    return app
