import re
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Standard tags that MediaWiki applies to reverted edits
MW_REVERTED_TAGS = {"mw-reverted"}

# Key revert patterns to search for in edit comments
# MediaWiki and common tools (Twinkle, Huggle, ClueBot) use these patterns
REVERT_PATTERNS = [
    r"revert(ed)?",
    r"undid",
    r"undo",
    r"rv\b",
    r"rollback",
    r"restored revision",
]


def clean_edit_comment(comment: str) -> str:
    """Removes standard MediaWiki section comments (e.g., /* Section name */) from the comment."""
    if not comment:
        return ""
    return re.sub(r"/\* .*? \*/", "", comment).strip()


def matches_revert_patterns(comment: str) -> bool:
    """Checks if the comment contains any typical revert keywords."""
    clean_c = clean_edit_comment(comment).lower()
    for pattern in REVERT_PATTERNS:
        if re.search(pattern, clean_c):
            return True
    return False


def is_comment_reverting_target(
    comment: str, target_user: str, target_revid: int
) -> bool:
    """
    Checks if a comment indicates a revert targeting a specific user or revision ID.
    """
    if not comment:
        return False

    clean_c = clean_edit_comment(comment).lower()

    # 1. If no revert keyword matches, it's not a revert comment
    if not matches_revert_patterns(clean_c):
        return False

    # 2. Check if the target user or target revision ID is mentioned in the comment
    target_user_lower = target_user.lower()
    revid_str = str(target_revid)

    user_pattern = r"\b" + re.escape(target_user_lower) + r"\b"
    revid_pattern = r"\b" + re.escape(revid_str) + r"\b"

    # Check for match of user or revid
    if re.search(revid_pattern, clean_c):
        logger.debug("Comment %r matches target revision ID %s", comment, revid_str)
        return True

    if target_user and re.search(user_pattern, clean_c):
        logger.debug("Comment %r matches target user %s", comment, target_user)
        return True

    return False


def check_revision_reverted(
    revision: Dict[str, Any], subsequent_revisions: List[Dict[str, Any]]
) -> bool:
    """
    Determines if the given revision was reverted.

    It checks:
    1. If the revision has 'mw-reverted' in its tags.
    2. If any of the subsequent revisions' comments indicate they reverted this revision.
    """
    # 1. Check tags
    tags = set(revision.get("tags", []))
    if tags & MW_REVERTED_TAGS:
        logger.info(
            "Revision %d is flagged as reverted via tag: %r",
            revision.get("revid"),
            tags & MW_REVERTED_TAGS,
        )
        return True

    # 2. Check subsequent comments
    target_user = revision.get("user", "")
    target_revid = revision.get("revid", 0)

    for follow_up in subsequent_revisions:
        comment = follow_up.get("comment", "")
        if is_comment_reverting_target(comment, target_user, target_revid):
            logger.info(
                "Revision %d (%s) was reverted by revision %d with comment: %r",
                target_revid,
                target_user,
                follow_up.get("revid"),
                comment,
            )
            return True

    return False
