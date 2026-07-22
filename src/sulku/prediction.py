"""
Prediction and Inference
========================

This module encapsulates all text classification and prediction logic,
including fastText model loading, Numpy 2.x patching, HMM forward-backward
smoothing, and sentence/paragraph scoring.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

import fasttext
import numpy as np

from sulku.constants import (
    DEFAULT_ALPHA,
    DEFAULT_P_STAY,
    LABEL_AI,
    LABEL_HUMAN,
    MODEL_PATHS,
    DEFAULT_LONG_PARAGRAPH_WORDS,
)
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
    predictions: Dict[str, float]
    confidences: Dict[str, float]
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

    def __init__(self) -> None:
        self.models: Dict[str, fasttext.FastText._FastText] = {}

    def load_models(self) -> None:
        """Load configured fastText models into memory."""
        for model_name, model_path in MODEL_PATHS.items():
            self.models[model_name] = fasttext.load_model(str(model_path.resolve().absolute()))

    def clear_models(self) -> None:
        """Clear all loaded fastText models from memory."""
        self.models.clear()

    @property
    def is_initialized(self) -> bool:
        """Check if any models are currently loaded."""
        return len(self.models) > 0

    def classify(
        self,
        text_or_paragraphs: str | List[Tuple[str, List[str]]],
        p_stay: float = DEFAULT_P_STAY,
        alpha: float = DEFAULT_ALPHA,
    ) -> EnsemblePredictionResult:
        """Classify a text document using the ensemble of loaded models.

        Accepts either a raw text string or pre-parsed paragraphs/sentences.
        Computes HMM-smoothed scores, weights results by word counts, and votes.
        """
        if not self.is_initialized:
            raise ValueError("Models not initialized.")

        if isinstance(text_or_paragraphs, str):
            parsed_paragraphs = parse_paragraphs_and_sentences(text_or_paragraphs)
        else:
            parsed_paragraphs = text_or_paragraphs

        if not parsed_paragraphs:
            raise ValueError("Text content does not contain classifiable paragraphs.")

        paragraph_sentences = [sentences for _, sentences in parsed_paragraphs]

        ai_votes = 0
        predictions = {}
        confidences = {}

        # Score each model concurrently to reduce end-to-end latency.
        with ThreadPoolExecutor(max_workers=max(1, len(self.models))) as executor:
            futures = {
                executor.submit(
                    _score_model,
                    name,
                    model,
                    paragraph_sentences,
                    p_stay=p_stay,
                    alpha=alpha,
                ): name
                for name, model in self.models.items()
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

            # If the averaged paragraph score is above 0.5, register an AI vote.
            if model_score > 0.5:
                ai_votes += 1

        final_score = float(np.mean(list(predictions.values()))) if predictions else 0.0
        final_confidence = float(np.mean(list(confidences.values()))) if confidences else 0.0
        logger.info("ensemble final_score=%.6f", final_score)
        logger.info("ensemble final_confidence=%.6f", final_confidence)

        # Majority voting logic (e.g., flagged if 2 or more models agree)
        is_ai = ai_votes >= 2

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

                # Calculate final_score as the mean of all models' predictions for this paragraph
                final_score_for_para = float(np.mean(list(predictions_for_para.values())))

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
            predictions=predictions,
            confidences=confidences,
            paragraphs=paragraphs_details,
        )


prediction_service = PredictionService()


