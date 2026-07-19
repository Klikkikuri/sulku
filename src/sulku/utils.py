"""
Text and Markdown Utilities
===========================

This module provides helper utilities for processing text and markdown content,
including stripping formatting and counting words.
"""

import re


def strip_markdown(text: str | None) -> str:
    """
    Strip markdown formatting from a text string, returning plain text.

    :param text: The markdown formatted string to clean.
    :type text: str | None
    :return: Cleaned plain text string.
    :rtype: str
    """
    if not text:
        return ""

    # 1. Strip YAML front matter
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            text = parts[2]

    # 2. Strip fenced code blocks (``` or ~~~)
    text = re.sub(r"```[^\n]*\n[\s\S]*?\n```", " ", text)
    text = re.sub(r"~~~[^\n]*\n[\s\S]*?\n~~~", " ", text)

    # 3. Strip inline code blocks
    text = re.sub(r"`{1,2}([^`]+)`{1,2}", r"\1", text)

    # 4. Remove image links (e.g. ![alt text](url) or ![alt text][ref])
    text = re.sub(r"!\[[^\]]*\]\((.*?)\)", " ", text)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]*\]", " ", text)

    # 5. Handle inline links: replace [link text](url) with link text
    text = re.sub(r"\[([^\]]+)\]\((.*?)\)", r"\1", text)

    # 6. Handle reference-style links: replace [link text][ref] with link text
    text = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", text)

    # 7. Remove link reference definitions like: [ref]: http://example.com
    text = re.sub(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", "", text, flags=re.MULTILINE)

    # 8. Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # 9. Strip headers formatting (remove leading # and trailing #)
    text = re.sub(r"^(?:#{1,6}\s+)(.*?)(?:\s*#*)?$", r"\1", text, flags=re.MULTILINE)

    # 10. Strip bold/italic/strikethrough formatting
    # Handle bold-italic (*** or ___), bold (** or __), italic (* or _) in order
    text = re.sub(r"\*\*\*([^*]+?)\*\*\*", r"\1", text)
    text = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+?)\*", r"\1", text)
    text = re.sub(r"___([^_]+?)___", r"\1", text)
    text = re.sub(r"__([^_]+?)__", r"\1", text)
    text = re.sub(r"_([^_]+?)_", r"\1", text)
    text = re.sub(r"~~([^~]+?)~~", r"\1", text)

    # 11. Remove blockquote markers (e.g., "> ")
    text = re.sub(r"^[ \t]*>[ \t]*", "", text, flags=re.MULTILINE)

    # 12. Remove list markers (e.g., "* ", "- ", "+ ", "1. ")
    text = re.sub(r"^[ \t]*[*+-][ \t]+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)

    # 13. Remove horizontal rules
    text = re.sub(
        r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "", text, flags=re.MULTILINE
    )

    # Replace pipes (tables) and backslashes (escaping)
    text = text.replace("|", " ")
    text = re.sub(r"\\(.)", r"\1", text)

    return text


def count_words(text: str | None) -> int:
    """
    Count the number of words in a markdown string after stripping formatting.

    :param text: The markdown formatted string to count.
    :type text: str | None
    :return: Word count.
    :rtype: int
    """
    cleaned = strip_markdown(text)
    words = cleaned.split()
    # Filter words to only count those with at least one alphanumeric character
    filtered_words = [w for w in words if any(c.isalnum() for c in w)]
    return len(filtered_words)
