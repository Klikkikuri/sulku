"""
Constants and Defaults
======================

This module defines the defaults and constant values used across the package.
"""

from pathlib import Path

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
DEFAULT_LONG_PARAGRAPH_WORDS = 15

