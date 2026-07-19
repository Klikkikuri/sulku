"""
Constants and Defaults
======================

This module defines the defaults and constant values used across the package.
"""

from pathlib import Path

# Dataset defaults
DEFAULT_SOURCE_DIR = Path("/app/data/yle")
DEFAULT_DEST_DIR_BASE = Path("/app/data/genai")

# Cache defaults
CACHE_APP_NAME = "sulku"
CACHE_APP_AUTHOR = "klikkikuri"
CACHE_SUBDIR = "summaries"

# Model defaults
DEFAULT_MODEL = "gemini-3.1-flash-lite"
