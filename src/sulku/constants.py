"""
Constants and Defaults
======================

This module defines the defaults and constant values used across the package.
"""

from pathlib import Path
from typing import Annotated, Dict, Tuple

DATA_DIR = Path("/app/data")

# Dataset defaults
DEFAULT_SOURCE_DIR = DATA_DIR / "yle"
DEFAULT_DEST_DIR_BASE = DATA_DIR / "genai"

# Cache defaults
CACHE_APP_NAME = "sulku"
CACHE_APP_AUTHOR = "klikkikuri"
CACHE_SUBDIR = "summaries"

# Model defaults
DEFAULT_MODEL = "gemini-3.1-flash-lite"
SUMMARIZE_MODEL = "gemini-3.1-flash-lite"

LABEL_AI = "__label__synthetic"
LABEL_HUMAN = "__label__human"

MODEL_PATHS = {
    "gemini-3.1-flash-lite": DATA_DIR / "models" / "gemini-3.1-flash-lite.ftz",
}

# HMM smoothing defaults
DEFAULT_P_STAY = 0.85
DEFAULT_ALPHA = 1.0

# Paragraph filtering defaults
DEFAULT_LONG_PARAGRAPH_WORDS = 10

# Voting defaults
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Z-score & Stouffer ensemble defaults
Mean = Annotated[float, "Mean prediction score on human baseline dataset"]
StdDev = Annotated[float, "Standard deviation of prediction scores on human baseline dataset"]

# Offline calibration parameters per model: model_name -> (mean, std)
MODEL_CALIBRATION: Dict[str, Tuple[Mean, StdDev]] = {}

# Fallback calibration parameters for uncalibrated models
DEFAULT_HUMAN_MEAN: float = 0.5
DEFAULT_HUMAN_STD: float = 0.10

# Stouffer combination decision threshold (Z-score cutoff for AI classification)
DEFAULT_Z_THRESHOLD: float = 1.0


