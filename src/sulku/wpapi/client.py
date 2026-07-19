import logging
import requests
import ratelimit
from typing import Any, Dict

logger = logging.getLogger(__name__)


class WikipediaAPIError(Exception):
    """Exception raised for errors in Wikipedia API response."""

    pass


class WikipediaClient:
    def __init__(
        self,
        default_lang: str = "en",
        user_agent: str = "SulkuPageFetcher/1.0 (klikkikuri@protonmail.com)",
    ):
        self.default_lang = default_lang
        self.user_agent = user_agent
        self.session = requests.Session()

    def get_api_url(self, lang: str) -> str:
        return f"https://{lang}.wikipedia.org/w/api.php"

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=5, period=1)
    def call_api(self, params: Dict[str, Any], lang: str = None) -> Dict[str, Any]:
        """
        Calls Wikipedia API with rate limits (max 5 calls per second).
        """
        lang = lang or self.default_lang
        url = self.get_api_url(lang)

        params.setdefault("format", "json")
        headers = {"User-Agent": self.user_agent}

        logger.debug("Calling Wikipedia API at %s with params: %r", url, params)
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("HTTP request to Wikipedia API failed: %s", e)
            raise WikipediaAPIError(f"HTTP request failed: {e}") from e

        data = response.json()
        if "error" in data:
            logger.error("Wikipedia API returned error: %r", data["error"])
            raise WikipediaAPIError(f"WPAPI Error: {data['error']}")

        return data


# Global client instance
wp_client = WikipediaClient()
