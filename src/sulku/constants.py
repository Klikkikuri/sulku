"""
Constants and Defaults
======================

This module defines the algorithm defaults and constant values used across the package.
Filesystem and deployment constants belong in ``sulku.bootstrap``.
"""

LABEL_AI = "__label__synthetic"
LABEL_HUMAN = "__label__human"

# HMM smoothing defaults
DEFAULT_P_STAY = 0.85
DEFAULT_ALPHA = 1.0

# Paragraph filtering defaults
DEFAULT_LONG_PARAGRAPH_WORDS = 10

# Voting defaults
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Z-score & Stouffer ensemble defaults
# Fallback calibration parameters for uncalibrated models
DEFAULT_HUMAN_MEAN: float = 0.5
DEFAULT_HUMAN_STD: float = 0.10

# Stouffer combination decision threshold (Z-score cutoff for AI classification)
DEFAULT_Z_THRESHOLD: float = 1.0
