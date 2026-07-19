import logging
from typing import Dict, Tuple, Optional
from .client import wp_client, WikipediaAPIError

logger = logging.getLogger(__name__)

# Simple in-memory cache for user revert statistics to avoid redundant calls in the same request/session
# Cache key: (username, lang, limit) -> (revert_rate, total_checked, reverted_count)
_user_trust_cache: Dict[Tuple[str, str, int], Tuple[Optional[float], int, int]] = {}


def clear_trust_cache():
    """Clears the user trust evaluation cache."""
    _user_trust_cache.clear()


def fetch_user_contributions(username: str, limit: int = 10, lang: str = "en") -> list:
    """
    Fetches the recent contributions for a given user.
    Only considers page edits (namespace 0).
    """
    params = {
        "action": "query",
        "list": "usercontribs",
        "ucuser": username,
        "uclimit": limit,
        "ucprop": "ids|title|timestamp|comment|tags|ns",
    }

    try:
        data = wp_client.call_api(params, lang=lang)
        contribs = data.get("query", {}).get("usercontribs", [])
        # Only return namespace 0 (articles)
        return [c for c in contribs if c.get("ns") == 0]
    except WikipediaAPIError as e:
        logger.error("Failed to fetch contributions for user %s: %s", username, e)
        return []


def compute_user_revert_rate(
    username: str, contrib_limit: int = 10, lang: str = "en", use_cache: bool = True
) -> Tuple[Optional[float], int, int]:
    """
    Computes the revert rate for a user by checking up to `contrib_limit` of their recent contributions.
    Returns a tuple of (revert_rate, total_checked, reverted_count).

    If the user has no contributions, returns (None, 0, 0).
    """
    cache_key = (username, lang, contrib_limit)
    if use_cache and cache_key in _user_trust_cache:
        logger.debug("Cache hit for user trust: %s", username)
        return _user_trust_cache[cache_key]

    contribs = fetch_user_contributions(username, limit=contrib_limit, lang=lang)
    if not contribs:
        res = (None, 0, 0)
        if use_cache:
            _user_trust_cache[cache_key] = res
        return res

    total = len(contribs)
    reverted = 0
    for contrib in contribs:
        tags = contrib.get("tags", [])
        if "mw-reverted" in tags:
            reverted += 1

    rate = reverted / total if total > 0 else 0.0
    logger.info(
        "User %s revert rate: %.2f%% (%d/%d)", username, rate * 100, reverted, total
    )

    res = (rate, total, reverted)
    if use_cache:
        _user_trust_cache[cache_key] = res
    return res


def is_user_trusted(
    username: str,
    max_revert_rate: float = 0.2,
    min_edits_for_evaluation: int = 3,
    contrib_limit: int = 10,
    lang: str = "en",
) -> Tuple[bool, Optional[float], str]:
    """
    Determines if a user is trusted based on their revert rate.

    Returns a tuple of:
    - is_trusted (bool)
    - revert_rate (Optional[float])
    - reason (str)
    """
    # IP edits / Anonymous users can be evaluated, but they are generally less trusted if they have any reverts.
    # We compute their revert rate.
    rate, total, reverted = compute_user_revert_rate(
        username, contrib_limit=contrib_limit, lang=lang
    )

    if rate is None or total == 0:
        # User has no history, treat neutrally
        return True, None, "User has no recent contributions to evaluate"

    if total < min_edits_for_evaluation:
        # Not enough edits to make a definitive judgment, but check if they have any reverts
        if reverted > 0:
            return (
                False,
                rate,
                f"User has too few edits ({total}) and some are reverted ({reverted})",
            )
        return True, rate, f"User has few edits ({total}) but none reverted"

    if rate >= max_revert_rate:
        return (
            False,
            rate,
            f"User revert rate ({rate:.2%}) exceeds threshold ({max_revert_rate:.2%})",
        )

    return True, rate, f"User revert rate ({rate:.2%}) is within safe limits"
