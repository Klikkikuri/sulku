from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
import fasttext
import logging
import numpy as np
from pydantic import BaseModel, Field
import re

# import fasttext
from sulku.wpapi import wpapi_router
from .constants import MODEL_PATHS
from sulku.utils import sentencize, strip_markdown

# Monkey-patch fasttext for NumPy 2.x compatibility
_orig_predict = fasttext.FastText._FastText.predict


def _patched_predict(self, text, k=1, threshold=0.0, on_unicode_error="strict"):
    try:
        return _orig_predict(self, text, k, threshold, on_unicode_error)
    except ValueError as e:
        if "Unable to avoid copy" in str(e):

            def check(entry):
                if entry.find("\n") != -1:
                    raise ValueError("predict processes one line at a time (remove '\\n')")
                entry += "\n"
                return entry

            if isinstance(text, list):
                text = [check(entry) for entry in text]
                all_labels, all_probs = self.f.multilinePredict(text, k, threshold, on_unicode_error)
                return all_labels, [np.asarray(p) for p in all_probs]
            else:
                text = check(text)
                predictions = self.f.predict(text, k, threshold, on_unicode_error)
                if predictions:
                    probs, labels = zip(*predictions)
                else:
                    probs, labels = ([], ())
                return labels, np.asarray(probs)
        raise


fasttext.FastText._FastText.predict = _patched_predict
logger = logging.getLogger(__name__)

# Global dictionary to hold pre-loaded models in memory
models: dict[str, fasttext.FastText._FastText] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models ONCE on server startup
    try:
        for model_name, model_path in MODEL_PATHS.items():
            models[model_name] = fasttext.load_model(str(model_path.resolve().absolute()))
        # models["deepseek"] = fasttext.load_model(MODEL_PATHS["deepseek"])
        # models["qwen"] = fasttext.load_model(MODEL_PATHS["qwen"])
        yield
    finally:
        # Clean up resources on shutdown if necessary
        models.clear()


class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=10, description="The document or sentence to analyze.")


class ClassificationResponse(BaseModel):
    is_ai: bool
    ai_votes: int
    total_models: int
    final_score: float
    predictions: dict[str, float]


router = APIRouter(prefix="/api/v1/aidetect", tags=["classification"])


def _paragraph_sentences(text: str, lang: str = "fi") -> list[list[str]]:
    """
    Build paragraph-level sentence units from text.

    Paragraphs are split by blank lines, then each paragraph is split into sentences.
    Sentences are whitespace-normalized so each unit can be safely passed to fastText
    without embedded newlines.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    paragraph_sentences: list[list[str]] = []

    for paragraph in raw_paragraphs:
        sentences = [" ".join(sentence.split()) for sentence in sentencize(paragraph, lang=lang) if sentence.strip()]
        if not sentences:
            fallback = " ".join(paragraph.split())
            if fallback:
                sentences = [fallback]

        if sentences:
            paragraph_sentences.append(sentences)

    return paragraph_sentences


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
async def classify_text(req: Request):
    if not models:
        raise HTTPException(status_code=500, detail="Models not initialized.")

    content_type = req.headers.get("content-type", "")
    main_type = content_type.split(";")[0].strip().lower() if content_type else "text/plain"

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
                detail=f"Unsupported Content-Type '{content_type}': Only text/plain, text/markdown, and application/json are allowed.",
            )

    # Apply minimum length validation on raw input
    if len(text) < 10:
        raise HTTPException(status_code=422, detail="Text content is too short (less than 10 characters).")

    # If it is markdown, strip formatting and frontmatter
    if is_markdown_content:
        text = strip_markdown(text).strip()
        if len(text) < 10:
            raise HTTPException(
                status_code=422,
                detail="Text content is too short (less than 10 characters) after stripping markdown and frontmatter.",
            )

    paragraph_sentences = _paragraph_sentences(text)
    if not paragraph_sentences:
        raise HTTPException(status_code=422, detail="Text content does not contain classifiable paragraphs.")

    ai_votes = 0
    predictions = {}

    # Process each paragraph through the ensemble by scoring each sentence first.
    for name, model in models.items():
        paragraph_scores: list[float] = []

        for idx, sentences in enumerate(paragraph_sentences, start=1):
            sentence_scores: list[float] = []

            for sentence in sentences:
                # fastText predict returns: (('__label__ai',), array([0.895]))
                labels, probabilities = model.predict(sentence)
                if not labels or len(probabilities) == 0:
                    continue

                label = labels[0].replace("__label__", "")
                prob = float(probabilities[0])
                sentence_scores.append(prob if label == "ai" else 1.0 - prob)

            if not sentence_scores:
                continue

            paragraph_score = float(np.mean(sentence_scores))
            paragraph_scores.append(paragraph_score)
            logger.info("model=%s paragraph=%d score=%.6f", name, idx, paragraph_score)

        if not paragraph_scores:
            raise HTTPException(status_code=500, detail=f"Model '{name}' returned no predictions.")

        model_score = float(np.mean(paragraph_scores))
        predictions[name] = model_score
        logger.info("model=%s final_score=%.6f", name, model_score)

        # If the averaged paragraph score is above 0.5, register an AI vote.
        if model_score > 0.5:
            ai_votes += 1

    final_score = float(np.mean(list(predictions.values()))) if predictions else 0.0
    logger.info("ensemble final_score=%.6f", final_score)

    # Majority voting logic (e.g., flagged if 2 or more models agree)
    is_ai = ai_votes >= 2

    return ClassificationResponse(
        is_ai=is_ai,
        ai_votes=ai_votes,
        total_models=len(models),
        final_score=final_score,
        predictions=predictions,
    )


def create_app() -> FastAPI:
    app = FastAPI(title="AI Text Classifier Service", lifespan=lifespan)

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    app.include_router(wpapi_router)
    app.include_router(router)

    return app
