from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.concurrency import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import fasttext
import logging
import numpy as np
from pydantic import BaseModel, Field
import re
import yaml

# import fasttext
from sulku.wpapi import wpapi_router
from .constants import LABEL_AI, MODEL_PATHS
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


def _extract_frontmatter_language(markdown_text: str) -> str | None:
    """
    Extract language from YAML front matter in markdown text.

    Supports both 'language' and 'lang' keys.
    """
    if not markdown_text.startswith("---"):
        return None

    parts = markdown_text.split("---", 2)
    if len(parts) < 3:
        return None

    yaml_part = parts[1].strip()
    try:
        parsed = yaml.safe_load(yaml_part)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    for key in ("language", "lang"):
        value = parsed.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized:
                return normalized

    return None


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


def _score_model(
    model_name: str,
    model: fasttext.FastText._FastText,
    paragraph_sentences: list[list[str]],
) -> tuple[str, float, list[float]]:
    """
    Score one model against paragraph->sentence inputs.

    Returns model name, final model score, and per-paragraph scores.
    """
    paragraph_scores: list[float] = []
    paragraph_weights: list[float] = []

    for sentences in paragraph_sentences:
        sentence_scores: list[float] = []
        paragraph_word_count = sum(len(sentence.split()) for sentence in sentences)

        for sentence in sentences:
            labels, probabilities = model.predict(sentence, k=1)
            if not labels or len(probabilities) == 0:
                continue

            label = labels[0]

            prob = float(probabilities[0])
            sentence_scores.append(prob if label == LABEL_AI else 1.0 - prob)

        if not sentence_scores:
            continue

        paragraph_scores.append(float(np.mean(sentence_scores)))
        paragraph_weights.append(float(max(1, paragraph_word_count)))

    if not paragraph_scores:
        raise ValueError(f"Model '{model_name}' returned no predictions.")

    model_score = float(np.average(paragraph_scores, weights=paragraph_weights))
    return model_name, model_score, paragraph_scores


@router.post("/", response_model=ClassificationResponse)
async def classify_text(req: Request):
    if not models:
        raise HTTPException(status_code=500, detail="Models not initialized.")

    content_type = req.headers.get("content-type", "")
    main_type = content_type.split(";")[0].strip().lower() if content_type else "text/plain"

    # Read raw body bytes to verify content
    body_bytes = await req.body()
    verify_plain_text(body_bytes)
    sentencize_lang = "fi"

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
        frontmatter_language = _extract_frontmatter_language(text)
        if frontmatter_language:
            sentencize_lang = frontmatter_language
        text = strip_markdown(text).strip()
        if len(text) < 10:
            raise HTTPException(
                status_code=422,
                detail="Text content is too short (less than 10 characters) after stripping markdown and frontmatter.",
            )

    paragraph_sentences = _paragraph_sentences(text, lang=sentencize_lang)
    if not paragraph_sentences:
        raise HTTPException(status_code=422, detail="Text content does not contain classifiable paragraphs.")

    ai_votes = 0
    predictions = {}

    # Score each model concurrently to reduce end-to-end latency.
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(models))) as executor:
            futures = {
                executor.submit(_score_model, name, model, paragraph_sentences): name for name, model in models.items()
            }
            model_results: list[tuple[str, float, list[float]]] = []

            for future in as_completed(futures):
                model_results.append(future.result())
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    for name, model_score, paragraph_scores in sorted(model_results, key=lambda item: item[0]):
        for idx, paragraph_score in enumerate(paragraph_scores, start=1):
            logger.info("model=%s paragraph=%d score=%.6f", name, idx, paragraph_score)

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
