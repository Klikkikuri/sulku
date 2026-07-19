from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional
import logging

from .service import get_wikipedia_page
from .client import WikipediaAPIError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/wikipedia", tags=["wikipedia"])


class WikipediaPageResponse(BaseModel):
    title: str = Field(..., description="The queried page title.")
    page_id: int = Field(..., description="The Wikipedia page ID.")
    revision_id: int = Field(..., description="The selected revision ID.")
    timestamp: str = Field(..., description="Timestamp of the selected revision.")
    editor: str = Field(..., description="The editor of the selected revision.")
    is_clean_evaluated: bool = Field(
        ..., description="Whether clean evaluation was executed."
    )
    editor_revert_rate: Optional[float] = Field(
        None, description="The revert rate of the editor (0.0 to 1.0)."
    )
    evaluation_notes: str = Field(
        ..., description="Details and logs of the revision scanning decision."
    )
    content: str = Field(
        ..., description="The content of the Wikipedia page in the requested format."
    )


@router.get("/page", response_model=WikipediaPageResponse)
def fetch_page(
    title: str = Query(..., description="The title of the Wikipedia page to fetch."),
    before_date: Optional[str] = Query(
        None,
        description="Fetch the revision before this date (ISO 8601 or YYYY-MM-DD).",
    ),
    prefer_clean: bool = Query(
        True,
        description="Whether to prefer a clean, non-vandalized revision by scanning recent edits and editor revert rates.",
    ),
    max_history_scan: int = Query(
        10,
        ge=1,
        le=50,
        description="The max number of recent revisions to evaluate for clean checks (1-50).",
    ),
    max_revert_rate: float = Query(
        0.2,
        ge=0.0,
        le=1.0,
        description="The maximum allowable revert rate for an editor to be trusted.",
    ),
    content_format: str = Query(
        "wikitext",
        pattern="^(wikitext|html|markdown)$",
        description="The content format to return ('wikitext', 'html', or 'markdown').",
    ),
    lang: str = Query(
        "en",
        min_length=2,
        max_length=10,
        description="The Wikipedia language code (e.g. 'en', 'fi', 'es').",
    ),
):
    """
    Fetches the content of a Wikipedia page.

    If `prefer_clean` is true, evaluates recent revisions for vandalism, reverts,
    and editor trust scores, selecting the latest stable revision.

    If `before_date` is provided, fetches the page revision state before that date.
    """
    try:
        page_data = get_wikipedia_page(
            title=title,
            before_date=before_date,
            prefer_clean=prefer_clean,
            max_history_scan=max_history_scan,
            max_revert_rate=max_revert_rate,
            content_format=content_format,
            lang=lang,
        )
        return page_data
    except WikipediaAPIError as e:
        error_msg = str(e)
        logger.warning("Wikipedia API error in endpoint: %s", error_msg)
        if "does not exist" in error_msg:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error_msg)
        elif "invalid" in error_msg.lower() or "format" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=error_msg
            )
    except Exception as e:
        logger.exception("Unexpected error in Wikipedia endpoint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {e}",
        )
