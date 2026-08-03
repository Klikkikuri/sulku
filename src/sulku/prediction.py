"""
Prediction and Inference
========================

This module encapsulates all text classification and prediction logic,including fastText model loading,
Numpy 2.x patching, HMM forward-backward smoothing, and sentence/paragraph scoring.

Detection Logic
---------------
Each loaded model is a specialist tuned to detect output from a particular AI generator (e.g. GPT-style, Claude-style,
Llama-style). Because a specialist model is expected to score near its human baseline on text from a generator it wasn't
trained on, a simple majority vote is unreliable.

Instead, we convert model scores to Z-scores relative to human baselines, clip uninformative negative Z-scores, and
combine them using Stouffer's method over a fixed denominator (sqrt(k)). This avoids selection bias while preventing
mismatched specialists from diluting correct positive detections.
"""

import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import time
from typing import Dict, List, Tuple

import fasttext
import numpy as np

from sulku.constants import (
    DEFAULT_ALPHA,
    DEFAULT_HUMAN_MEAN,
    DEFAULT_HUMAN_STD,
    DEFAULT_LONG_PARAGRAPH_WORDS,
    DEFAULT_P_STAY,
    DEFAULT_Z_THRESHOLD,
    LABEL_AI,
    LABEL_HUMAN,
    MODEL_CALIBRATION,
    MODEL_PATHS,
)
from sulku.metrics import model_in_flight, model_loads, model_unloads
from sulku.utils import parse_paragraphs_and_sentences

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
_load_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model_loader")


@dataclass
class ParagraphPredictionDetail:
    """Dataclass holding predictions and metadata for a single paragraph."""

    text: str
    sentences: List[str]
    predictions: Dict[str, float] | None
    final_score: float | None


@dataclass
class EnsemblePredictionResult:
    """Dataclass holding the structured result of an ensemble classification."""

    is_ai: bool
    ai_votes: int
    total_models: int
    final_score: float
    final_confidence: float
    final_z_score: float
    predictions: Dict[str, float]
    confidences: Dict[str, float]
    z_scores: Dict[str, float]
    paragraphs: List[ParagraphPredictionDetail]


def forward_backward_smoothing(
    p_ai_list: List[float],
    p_human_list: List[float],
    p_stay: float = DEFAULT_P_STAY,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[List[float], List[float]]:
    """Apply Forward-Backward HMM-style smoothing to sentence scores.

    Treats AI (state 0) and Human (state 1) as hidden states.
    The transition matrix is defined by:
        P(stay) = p_stay
        P(switch) = 1 - p_stay
    The per-sentence fastText scores are treated as emission probabilities,
    scaled by parameter alpha (i.e. emission^alpha).
    """
    n = len(p_ai_list)
    if n == 0:
        return [], []

    # State 0: AI, State 1: Human
    p_switch = 1.0 - p_stay
    A = np.array([[p_stay, p_switch], [p_switch, p_stay]])

    emissions = np.zeros((n, 2))
    for t in range(n):
        emissions[t, 0] = max(p_ai_list[t], 1e-12) ** alpha
        emissions[t, 1] = max(p_human_list[t], 1e-12) ** alpha

    pi = np.array([0.5, 0.5])

    # Forward pass
    alpha_f = np.zeros((n, 2))
    c = np.zeros(n)

    alpha_f[0] = pi * emissions[0]
    c[0] = np.sum(alpha_f[0])
    if c[0] > 0:
        alpha_f[0] /= c[0]
    else:
        alpha_f[0] = np.array([0.5, 0.5])
        c[0] = 1.0

    for t in range(1, n):
        alpha_f[t] = emissions[t] * (alpha_f[t - 1] @ A)
        c[t] = np.sum(alpha_f[t])
        if c[t] > 0:
            alpha_f[t] /= c[t]
        else:
            alpha_f[t] = np.array([0.5, 0.5])
            c[t] = 1.0

    # Backward pass
    beta_b = np.zeros((n, 2))
    beta_b[-1] = np.array([1.0, 1.0])

    for t in range(n - 2, -1, -1):
        beta_b[t] = (A @ (emissions[t + 1] * beta_b[t + 1])) / c[t + 1]

    # Posterior marginals
    gamma = alpha_f * beta_b
    row_sums = np.sum(gamma, axis=1, keepdims=True)
    gamma = np.where(row_sums > 0, gamma / row_sums, np.array([0.5, 0.5]))

    smoothed_p_ai = [float(val) for val in gamma[:, 0]]
    smoothed_p_human = [float(val) for val in gamma[:, 1]]

    return smoothed_p_ai, smoothed_p_human


def _score_model(
    model_name: str,
    model: fasttext.FastText._FastText,
    paragraph_sentences: List[List[str]],
    p_stay: float = DEFAULT_P_STAY,
    alpha: float = DEFAULT_ALPHA,
) -> Tuple[str, float, List[float], float]:
    """Score one model against paragraph->sentence inputs.

    Confidence is calculated as the margin between the AI and human probabilities.
    """
    # Keep paragraphs that have at least DEFAULT_LONG_PARAGRAPH_WORDS words in total
    eligible_paragraphs = [
        sentences for sentences in paragraph_sentences
        if sum(len(s.split()) for s in sentences) >= DEFAULT_LONG_PARAGRAPH_WORDS
    ]
    filtered_paragraphs = eligible_paragraphs if eligible_paragraphs else paragraph_sentences

    paragraph_scores: List[float] = []
    paragraph_weights: List[float] = []
    paragraph_margins: List[float] = []

    for sentences in filtered_paragraphs:
        raw_p_ai: List[float] = []
        raw_p_human: List[float] = []
        sentence_weights: List[float] = []

        for sentence in sentences:
            labels, probabilities = model.predict(sentence, k=2)  # request both labels
            if not labels or len(probabilities) == 0:
                continue

            # Build a label -> probability map since fastText doesn't guarantee order
            label_probs = dict(zip(labels, (float(p) for p in probabilities)))

            p_ai = label_probs.get(LABEL_AI, 0.0)
            # If the model only returned one label (rare, but possible on short/degenerate input),
            # treat the missing one as the complement.
            p_human = label_probs.get(LABEL_HUMAN, 1.0 - p_ai) if len(label_probs) > 1 else 1.0 - p_ai

            raw_p_ai.append(p_ai)
            raw_p_human.append(p_human)
            sentence_weights.append(float(max(1, len(sentence.split()))))

        if not raw_p_ai:
            continue

        # Run forward-backward smoothing on raw scores before paragraph collapse
        smoothed_p_ai, smoothed_p_human = forward_backward_smoothing(
            raw_p_ai, raw_p_human, p_stay=p_stay, alpha=alpha
        )

        # Recompute smoothed margins
        sentence_margins = [abs(ai - hum) for ai, hum in zip(smoothed_p_ai, smoothed_p_human)]

        paragraph_scores.append(float(np.average(smoothed_p_ai, weights=sentence_weights)))
        paragraph_weights.append(float(sum(sentence_weights)))
        paragraph_margins.append(float(np.average(sentence_margins, weights=sentence_weights)))

    if not paragraph_scores:
        raise ValueError(f"Model '{model_name}' returned no predictions.")

    model_score = float(np.average(paragraph_scores, weights=paragraph_weights))
    model_confidence = float(np.average(paragraph_margins, weights=paragraph_weights))
    return model_name, model_score, paragraph_scores, model_confidence


class PredictionService:
    """Service to load fastText models and run ensemble text classification."""

    default_keep_alive: float = 300.0
    min_loaded_time: float = 60.0  # cooldown — no eviction within N s of load

    def __init__(self) -> None:
        self.models: Dict[str, fasttext.FastText._FastText] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._last_used: Dict[str, float] = {}
        self._loaded_at: Dict[str, float] = {}
        self._in_flight: Dict[str, int] = {}
        self._keep_alive: Dict[str, float] = {}  # per-model TTL overrides
        self._inter_request_sum: Dict[str, float] = {}
        self._inter_request_n: Dict[str, int] = {}
        self._eviction_task: asyncio.Task | None = None

    def _update_adaptive_ttl(self, name: str, elapsed_since_last: float) -> None:
        """Adjust per-model keep_alive toward k × mean inter-request interval (k=3).

        Writes into ``self._keep_alive[name]`` (per-model dict) so models with
        different traffic patterns each maintain an independent TTL. The single
        scalar ``self.default_keep_alive`` is never mutated here — it stays as
        the fallback for newly-loaded models.
        """
        k = 3.0
        min_ttl, max_ttl = 60.0, 1800.0
        n = self._inter_request_n.get(name, 0) + 1
        s = self._inter_request_sum.get(name, 0.0) + elapsed_since_last
        self._inter_request_n[name] = n
        self._inter_request_sum[name] = s
        adaptive = max(min_ttl, min(max_ttl, k * (s / n)))
        prev = self._keep_alive.get(name, self.default_keep_alive)
        if abs(adaptive - prev) > 10:
            logger.debug(
                "adaptive TTL %.0f→%.0f s model=%s (n=%d samples)",
                prev, adaptive, name, n,
            )
            self._keep_alive[name] = adaptive

    def get_keep_alive(self, name: str) -> float:
        """Return the effective TTL for *name*, falling back to the default."""
        return self._keep_alive.get(name, self.default_keep_alive)

    def evict_model(self, name: str, reason: str = "eviction") -> None:
        """Unload model from memory by deleting the Python reference."""
        if name not in self.models:
            return
        del self.models[name]
        self._last_used.pop(name, None)
        self._loaded_at.pop(name, None)
        self._keep_alive.pop(name, None)
        self._in_flight.pop(name, None)
        model_unloads.add(1, {"model": name, "reason": reason})
        logger.info("model=%s evicted reason=%s (~92 MB heap freed)", name, reason)

    def unload_model(self, name: str) -> None:
        """Manual eviction via API; raises KeyError if model not loaded."""
        if name not in self.models:
            raise KeyError(name)
        self.evict_model(name, reason="manual")

    async def ensure_loaded(self, name: str) -> None:
        """Ensure a specific model by name is loaded into memory."""
        now = time.monotonic()
        if name in self.models:
            if name in self._last_used:
                elapsed = now - self._last_used[name]
                self._update_adaptive_ttl(name, elapsed)
            self._last_used[name] = now
            return
        if name not in MODEL_PATHS:
            raise KeyError(f"Unknown model '{name}'.")
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        async with self._locks[name]:
            if name in self.models:  # double-checked
                if name in self._last_used:
                    elapsed = now - self._last_used[name]
                    self._update_adaptive_ttl(name, elapsed)
                self._last_used[name] = now
                return
            path = MODEL_PATHS[name]
            loop = asyncio.get_running_loop()
            model = await loop.run_in_executor(
                _load_executor, fasttext.load_model, str(path.resolve().absolute())
            )
            self.models[name] = model
            self._last_used[name] = now
            self._loaded_at[name] = now
            self._in_flight[name] = 0
            model_loads.add(1, {"model": name})
            logger.info("model=%s loaded", name)

    @contextlib.contextmanager
    def track_in_flight(self, name: str):
        """Context manager to track active in-flight requests per model."""
        self._in_flight[name] = self._in_flight.get(name, 0) + 1
        model_in_flight.add(1, {"model": name})
        try:
            yield
        finally:
            self._in_flight[name] = max(0, self._in_flight.get(name, 1) - 1)
            model_in_flight.add(-1, {"model": name})

    def load_models(self) -> None:
        """Load configured fastText models into memory."""
        now = time.monotonic()
        for model_name, model_path in MODEL_PATHS.items():
            if model_name not in self.models:
                self.models[model_name] = fasttext.load_model(str(model_path.resolve().absolute()))
                self._last_used[model_name] = now
                self._loaded_at[model_name] = now
                self._in_flight[model_name] = 0
                model_loads.add(1, {"model": model_name})

    def clear_models(self) -> None:
        """Clear all loaded fastText models from memory."""
        for name in list(self.models.keys()):
            self.evict_model(name, reason="manual")

    @property
    def is_initialized(self) -> bool:
        """Check if any models are currently loaded."""
        return len(self.models) > 0


    def classify(
        self,
        text_or_paragraphs: str | List[Tuple[str, List[str]]],
        p_stay: float = DEFAULT_P_STAY,
        alpha: float = DEFAULT_ALPHA,
        models: List[str] | None = None,
    ) -> EnsemblePredictionResult:
        """Classify a text document using the ensemble of loaded models.

        Accepts either a raw text string or pre-parsed paragraphs/sentences.
        Computes HMM-smoothed scores, converts to Z-scores, clips negative Z-scores,
        and combines scores using Stouffer's formula with a fixed denominator.
        """
        if not self.is_initialized:
            raise ValueError("Models not initialized.")

        target_models = self.models
        if models is not None:
            # Filter models to specified subset
            missing = [m for m in models if m not in self.models]
            if missing:
                raise ValueError(f"Requested model(s) not found/loaded: {', '.join(missing)}")
            if not models:
                raise ValueError("At least one model must be specified in models parameter.")
            target_models = {m: self.models[m] for m in models}

        if isinstance(text_or_paragraphs, str):
            parsed_paragraphs = parse_paragraphs_and_sentences(text_or_paragraphs)
        else:
            parsed_paragraphs = text_or_paragraphs

        if not parsed_paragraphs:
            raise ValueError("Text content does not contain classifiable paragraphs.")

        paragraph_sentences = [sentences for _, sentences in parsed_paragraphs]

        predictions = {}
        confidences = {}

        # Score each model concurrently to reduce end-to-end latency.
        with ThreadPoolExecutor(max_workers=max(1, len(target_models))) as executor:
            def _score_with_tracking(m_name: str, m_obj: fasttext.FastText._FastText):
                self._last_used[m_name] = time.monotonic()
                with self.track_in_flight(m_name):
                    return _score_model(
                        m_name,
                        m_obj,
                        paragraph_sentences,
                        p_stay=p_stay,
                        alpha=alpha,
                    )

            futures = {
                executor.submit(
                    _score_with_tracking,
                    name,
                    model,
                ): name
                for name, model in target_models.items()
            }
            model_results: List[Tuple[str, float, List[float], float]] = []

            for future in as_completed(futures):
                model_results.append(future.result())


        for name, model_score, paragraph_scores, model_confidence in sorted(model_results, key=lambda item: item[0]):
            for idx, paragraph_score in enumerate(paragraph_scores, start=1):
                logger.info("model=%s paragraph=%d score=%.6f", name, idx, paragraph_score)

            predictions[name] = model_score
            confidences[name] = model_confidence
            logger.info("model=%s final_score=%.6f", name, model_score)
            logger.info("model=%s confidence=%.6f", name, model_confidence)

        # Convert predictions to Z-scores relative to human baselines
        z_scores: Dict[str, float] = {}
        clipped_z_sum = 0.0

        for name, model_score in sorted(predictions.items()):
            mean, std = MODEL_CALIBRATION.get(name, (DEFAULT_HUMAN_MEAN, DEFAULT_HUMAN_STD))
            std_val = std if std > 1e-6 else 1e-6
            z = (model_score - mean) / std_val
            z_scores[name] = float(z)
            # Clip negative z-scores to 0 (uninformative baseline readings)
            clipped_z_sum += max(0.0, float(z))

        total_models = max(1, len(self.models))
        # Fixed denominator (sqrt(total_models)) prevents selection bias
        final_z_score = float(clipped_z_sum / (total_models ** 0.5)) if predictions else 0.0
        final_score = float(np.mean(list(predictions.values()))) if predictions else 0.0
        final_confidence = float(np.mean(list(confidences.values()))) if confidences else 0.0
        is_ai = final_z_score >= DEFAULT_Z_THRESHOLD
        ai_votes = sum(1 for z in z_scores.values() if z > 0.0)

        # TODO: Generalist Fallback Tier Design
        # When no specialist model stands out (final_z_score < DEFAULT_Z_THRESHOLD), query a generalist model
        # trained on pooled AI text across all known generators vs. human text instead of majority voting
        # among mismatched specialists that failed to agree. If generalist_score >= GENERALIST_THRESHOLD,
        # set is_ai = True with fallback_triggered = True.

        logger.info("ensemble final_z_score=%.6f", final_z_score)
        logger.info("ensemble final_score=%.6f", final_score)
        logger.info("ensemble final_confidence=%.6f", final_confidence)

        # Map model paragraph-level predictions back to the original parsed paragraphs
        eligible_indices = [
            i for i, sentences in enumerate(paragraph_sentences)
            if sum(len(s.split()) for s in sentences) >= DEFAULT_LONG_PARAGRAPH_WORDS
        ]
        if eligible_indices:
            classified_indices = set(eligible_indices)
        else:
            classified_indices = set(range(len(paragraph_sentences)))

        classified_indices_list = sorted(list(classified_indices))

        paragraphs_details: List[ParagraphPredictionDetail] = []
        for i, (para_text, sentences) in enumerate(parsed_paragraphs):
            if i in classified_indices:
                k = classified_indices_list.index(i)
                # Map predictions per model for this paragraph
                predictions_for_para = {}
                for name, _, p_scores, _ in model_results:
                    predictions_for_para[name] = p_scores[k]

                final_score_for_para = float(np.mean(list(predictions_for_para.values()))) if predictions_for_para else 0.0

                paragraphs_details.append(
                    ParagraphPredictionDetail(
                        text=para_text,
                        sentences=sentences,
                        predictions=predictions_for_para,
                        final_score=final_score_for_para,
                    )
                )
            else:
                paragraphs_details.append(
                    ParagraphPredictionDetail(
                        text=para_text,
                        sentences=sentences,
                        predictions=None,
                        final_score=None,
                    )
                )

        return EnsemblePredictionResult(
            is_ai=is_ai,
            ai_votes=ai_votes,
            total_models=len(self.models),
            final_score=final_score,
            final_confidence=final_confidence,
            final_z_score=final_z_score,
            predictions=predictions,
            confidences=confidences,
            z_scores=z_scores,
            paragraphs=paragraphs_details,
        )



prediction_service = PredictionService()


