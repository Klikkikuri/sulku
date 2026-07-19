import logging
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple

from .client import wp_client, WikipediaClient, WikipediaAPIError
from .revert import check_revision_reverted
from .trust import is_user_trusted, clear_trust_cache

logger = logging.getLogger(__name__)


class WikipediaPageService:
    """
    Service for retrieving and evaluating Wikipedia page content.
    Supports fetching revisions before a specific date and selecting
    the latest clean/non-vandalized version by scanning page history.
    Supports converting page HTML content to clean Markdown.
    """

    def __init__(self, client: Optional[WikipediaClient] = None):
        self.client = client or wp_client

    def parse_iso_datetime(self, dt_str: str) -> datetime:
        """Parses ISO 8601 datetime strings, handling 'Z' suffix for UTC."""
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.fromisoformat(dt_str)

    def format_to_wp_timestamp(self, date_str: Optional[str]) -> Optional[str]:
        """
        Parses common date/datetime formats and returns Wikipedia's expected
        ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ.
        """
        if not date_str:
            return None

        date_str = date_str.strip()

        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            # Include the whole day by default
            return f"{date_str}T23:59:59Z"

        try:
            dt = self.parse_iso_datetime(date_str)
            if dt.tzinfo:
                dt = dt.astimezone(timezone.utc)
            else:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError as e:
            logger.warning(
                "Could not parse date string %r: %s. Using as-is.", date_str, e
            )
            return date_str

    def fetch_revision_history(
        self,
        title: str,
        limit: int = 15,
        before_timestamp: Optional[str] = None,
        lang: str = "en",
    ) -> Tuple[List[Dict[str, Any]], Optional[int]]:
        """
        Fetches the revision history of a page.
        """
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": limit,
            "rvprop": "ids|timestamp|user|comment|tags",
        }

        if before_timestamp:
            params["rvstart"] = before_timestamp
            params["rvdir"] = "older"

        data = self.client.call_api(params, lang=lang)
        pages = data.get("query", {}).get("pages", {})

        for page_id, page in pages.items():
            if "missing" in page:
                raise WikipediaAPIError(f"The page '{title}' does not exist.")
            if "invalid" in page:
                raise WikipediaAPIError(f"The page title '{title}' is invalid.")

            revisions = page.get("revisions", [])
            return revisions, int(page_id)

        return [], None

    def fetch_revisions_after(
        self, title: str, limit: int = 5, start_timestamp: str = "", lang: str = "en"
    ) -> List[Dict[str, Any]]:
        """
        Fetches revisions starting from a specific timestamp and going forward in time.
        """
        if not start_timestamp:
            return []

        try:
            dt = self.parse_iso_datetime(start_timestamp)
            start_timestamp_adjusted = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            start_timestamp_adjusted = start_timestamp

        params = {
            "action": "query",
            "prop": "revisions",
            "titles": title,
            "rvlimit": limit,
            "rvstart": start_timestamp_adjusted,
            "rvdir": "newer",
            "rvprop": "ids|timestamp|user|comment|tags",
        }

        try:
            data = self.client.call_api(params, lang=lang)
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "missing" not in page:
                    return page.get("revisions", [])
        except Exception as e:
            logger.warning("Failed to fetch newer revisions for revert check: %s", e)

        return []

    def mediawiki_html_to_markdown(self, html_content: str, lang: str = "en") -> str:
        """
        Converts Wikipedia HTML content to clean Markdown, applying
        Wikipedia-specific cleaning heuristics inspired by the meri codebase.
        """
        from copy import deepcopy
        from trafilatura import load_html
        from trafilatura.core import Extractor
        from trafilatura.htmlprocessing import convert_tags, prune_unwanted_nodes
        from trafilatura.xml import xmltotxt
        from lxml.etree import XPath
        from trafilatura.settings import DEFAULT_CONFIG

        config = deepcopy(DEFAULT_CONFIG)
        # Avoid warnings by ensuring User-Agent configuration
        config["DEFAULT"].setdefault("USER_AGENTS", self.client.user_agent)

        _extractor_args = {
            "config": config,
            "output_format": "markdown",
            "formatting": True,
            "links": True,
            "images": False,
            "tables": True,
            "comments": False,
        }

        html = load_html(html_content)
        if html is None:
            raise WikipediaAPIError(
                "Failed to parse HTML document for markdown conversion."
            )

        # Prune common layout and form elements
        html = prune_unwanted_nodes(
            html,
            [
                XPath(x)
                for x in [
                    "//script",
                    "//noscript",
                    "//style",
                    "//link",
                    "//meta",
                    "//form",
                    "//input",
                    "//button",
                ]
            ],
        )

        # Wikipedia-specific cleanup:
        # Redirect links, new pages that don't exist, print-only elements, and notices
        html = prune_unwanted_nodes(
            html,
            [
                XPath(x)
                for x in [
                    '//a[contains(@class, "mw-redirect") or contains(@class, "new")]',
                    '//*[contains(@class, "noprint") or contains(@class, "ambox-notice")]',
                ]
            ],
        )

        options = Extractor(**_extractor_args)
        html = convert_tags(html, options)
        txt = xmltotxt(html.body, options.formatting)

        if not txt:
            return ""

        # Post-processing cleanup:
        # 1. Remove citation links (e.g. [1], [23])
        txt = re.sub(r"\[\s*\d+\]", "", txt)
        # 2. Clean whitespace before word boundaries, left by inline tags (e.g., <i> <b>)
        txt = re.sub(r"\s+([.,;:!?)])", r"\1", txt)

        return txt.strip()

    def fetch_revision_content(
        self,
        page_title: str,
        revid: int,
        content_format: str = "wikitext",
        lang: str = "en",
    ) -> str:
        """
        Fetches the content of a specific revision.
        Supports formats: 'wikitext', 'html', or 'markdown'.
        """
        if content_format in ("html", "markdown"):
            params = {
                "action": "parse",
                "oldid": revid,
                "prop": "text",
            }
            data = self.client.call_api(params, lang=lang)
            html_content = data.get("parse", {}).get("text", {}).get("*", "")

            if content_format == "markdown":
                return self.mediawiki_html_to_markdown(html_content, lang=lang)
            return html_content
        else:
            params = {
                "action": "query",
                "prop": "revisions",
                "revids": revid,
                "rvprop": "content",
            }
            data = self.client.call_api(params, lang=lang)
            pages = data.get("query", {}).get("pages", {})
            for _, page in pages.items():
                revisions = page.get("revisions", [])
                if revisions:
                    return revisions[0].get("*", "")
            raise WikipediaAPIError(f"Content for revision {revid} not found.")

    def get_wikipedia_page(
        self,
        title: str,
        before_date: Optional[str] = None,
        prefer_clean: bool = True,
        max_history_scan: int = 10,
        max_revert_rate: float = 0.2,
        content_format: str = "wikitext",
        lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Retrieves the Wikipedia page content, applying clean version detection and date filtering.
        """
        wp_before_timestamp = self.format_to_wp_timestamp(before_date)

        # Clear trust cache to get fresh trust evaluations
        clear_trust_cache()

        revisions, page_id = self.fetch_revision_history(
            title,
            limit=max_history_scan,
            before_timestamp=wp_before_timestamp,
            lang=lang,
        )

        if not revisions:
            raise WikipediaAPIError(f"No revisions found for page '{title}'.")

        page_id = page_id or 0

        newer_revisions = []
        if wp_before_timestamp:
            newer_revisions = self.fetch_revisions_after(
                title, limit=5, start_timestamp=wp_before_timestamp, lang=lang
            )

        all_known_revisions = newer_revisions + revisions

        selected_rev = None
        evaluation_notes = []

        if not prefer_clean:
            selected_rev = revisions[0]
            evaluation_notes.append(
                "Page fetched without clean evaluation (prefer_clean=False)."
            )
            editor_revert_rate = None
        else:
            fallback_rev = revisions[0]
            fallback_non_reverted_rev = None

            for idx, rev in enumerate(revisions):
                revid = rev.get("revid")
                user = rev.get("user", "")

                newer_than_rev = (
                    all_known_revisions[: all_known_revisions.index(rev)]
                    if rev in all_known_revisions
                    else all_known_revisions[:idx]
                )

                is_reverted = check_revision_reverted(rev, newer_than_rev)
                if is_reverted:
                    evaluation_notes.append(
                        f"Revision {revid} by '{user}' was bypassed (reverted)."
                    )
                    continue

                if not fallback_non_reverted_rev:
                    fallback_non_reverted_rev = rev

                is_trusted, revert_rate, trust_reason = is_user_trusted(
                    user, max_revert_rate=max_revert_rate, lang=lang
                )

                if not is_trusted:
                    evaluation_notes.append(
                        f"Revision {revid} by '{user}' was bypassed (untrusted editor: {trust_reason})."
                    )
                    continue

                selected_rev = rev
                editor_revert_rate = revert_rate
                evaluation_notes.append(
                    f"Selected revision {revid} by '{user}' as the latest clean version. Editor trust evaluation: {trust_reason}."
                )
                break

            if not selected_rev:
                if fallback_non_reverted_rev:
                    selected_rev = fallback_non_reverted_rev
                    _, r_rate, _ = is_user_trusted(
                        selected_rev.get("user", ""), lang=lang
                    )
                    editor_revert_rate = r_rate
                    evaluation_notes.append(
                        f"Fallback to latest non-reverted revision {selected_rev.get('revid')} by '{selected_rev.get('user')}', "
                        "since no revision passed both clean and editor trust criteria."
                    )
                else:
                    selected_rev = fallback_rev
                    _, r_rate, _ = is_user_trusted(
                        selected_rev.get("user", ""), lang=lang
                    )
                    editor_revert_rate = r_rate
                    evaluation_notes.append(
                        f"Fallback to latest available revision {selected_rev.get('revid')} by '{selected_rev.get('user')}', "
                        "since all checked revisions are reverted."
                    )

        revid = selected_rev["revid"]
        content = self.fetch_revision_content(
            title, revid, content_format=content_format, lang=lang
        )

        return {
            "title": title,
            "page_id": page_id,
            "revision_id": revid,
            "timestamp": selected_rev.get("timestamp"),
            "editor": selected_rev.get("user"),
            "is_clean_evaluated": prefer_clean,
            "editor_revert_rate": editor_revert_rate,
            "evaluation_notes": "; ".join(evaluation_notes),
            "content": content,
        }


# Module-level convenience function
def get_wikipedia_page(
    title: str,
    before_date: Optional[str] = None,
    prefer_clean: bool = True,
    max_history_scan: int = 10,
    max_revert_rate: float = 0.2,
    content_format: str = "wikitext",
    lang: str = "en",
) -> Dict[str, Any]:
    """
    Convenience function to fetch a Wikipedia page using the default service.
    """
    service = WikipediaPageService()
    return service.get_wikipedia_page(
        title=title,
        before_date=before_date,
        prefer_clean=prefer_clean,
        max_history_scan=max_history_scan,
        max_revert_rate=max_revert_rate,
        content_format=content_format,
        lang=lang,
    )
