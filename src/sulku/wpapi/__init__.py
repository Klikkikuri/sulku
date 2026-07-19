"""
Wikipedia Page Fetching API Module.

This module provides services and endpoints to fetch Wikipedia page revisions,
with options to:
1. Fetch the revision of a page before a specific date.
2. Prefer clean/non-vandalized versions of a page by scanning recent history
   and evaluating both revision reverts (via MediaWiki tags and comment-based matching)
   and editor trustworthiness (calculating editor revert rates).

Public API:
- `WikipediaPageService`: Service class to orchestrate Wikipedia page fetching and evaluation.
- `get_wikipedia_page`: Convenience function to fetch and evaluate page revisions.
- `wpapi_router`: FastAPI router containing the page fetch endpoint.
"""

from .router import router as wpapi_router
from .service import get_wikipedia_page, WikipediaPageService

__all__ = ["wpapi_router", "get_wikipedia_page", "WikipediaPageService"]
