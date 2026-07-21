"""
Unit tests for text and markdown utilities
==========================================

Tests the strip_markdown and count_words helper functions under various
formatting and structure styles, ensuring correct output and count.
"""

from sulku.utils import count_words, sentencize, strip_markdown, is_markdown


def test_strip_markdown_none_and_empty():
    """Test that None and empty string inputs are handled correctly."""
    assert strip_markdown(None) == ""
    assert strip_markdown("") == ""
    assert count_words(None) == 0
    assert count_words("") == 0


def test_strip_markdown_plain_text():
    """Test plain text with no formatting."""
    text = "Hello world from python"
    assert strip_markdown(text) == text
    assert count_words(text) == 4


def test_strip_markdown_headers():
    """Test headers of different levels."""
    text = "# Header 1\n## Header 2\n### Header 3"
    assert strip_markdown(text) == "Header 1\nHeader 2\nHeader 3"
    assert count_words(text) == 6


def test_strip_markdown_emphasis():
    """Test bold, italic, and strikethrough markdown styles."""
    text = "This is **bold** text and *italic* and ~~strikethrough~~."
    expected = "This is bold text and italic and strikethrough."
    assert strip_markdown(text) == expected
    assert count_words(text) == 8


def test_strip_markdown_code_blocks():
    """Test inline and fenced code blocks."""
    inline_text = "Use the `count_words` function."
    assert strip_markdown(inline_text) == "Use the count_words function."
    assert count_words(inline_text) == 4

    fenced_text = "Before block\n```python\ndef foo():\n    return 42\n```\nAfter block"
    assert strip_markdown(fenced_text).strip() == "Before block\n \nAfter block"
    assert count_words(fenced_text) == 4


def test_strip_markdown_links():
    """Test links and image links."""
    links_text = "Go to [Google](https://google.com) or [GitHub](https://github.com)."
    assert strip_markdown(links_text) == "Go to Google or GitHub."
    assert count_words(links_text) == 5

    image_text = "See this ![cool logo](logo.png) now."
    assert strip_markdown(image_text) == "See this   now."
    assert count_words(image_text) == 3


def test_strip_markdown_reference_links():
    """Test reference-style links and reference definitions."""
    text = "Use [this link][1] and [another one][2].\n\n[1]: http://google.com\n[2]: http://github.com"
    expected = "Use this link and another one.\n\n\n"
    assert strip_markdown(text) == expected
    assert count_words(text) == 6


def test_strip_markdown_html():
    """Test inline HTML removal."""
    text = "<div>Hello <strong>world</strong>!</div>"
    assert strip_markdown(text).strip() == "Hello  world !"
    assert count_words(text) == 2


def test_strip_markdown_lists_and_quotes():
    """Test lists and blockquote formatting."""
    text = "> A quote here\n\n* Item 1\n- Item 2\n+ Item 3\n1. Numbered Item"
    expected = "A quote here\n\nItem 1\nItem 2\nItem 3\nNumbered Item"
    assert strip_markdown(text) == expected
    assert count_words(text) == 11


def test_strip_markdown_tables():
    """Test tables formatting is cleaned up."""
    text = "| Col 1 | Col 2 |\n|---|---|\n| Val A | Val B |"
    assert count_words(text) == 8


def test_strip_markdown_horizontal_rules():
    """Test horizontal rule markdown formats."""
    text = "Line 1\n---\nLine 2\n***\nLine 3"
    expected = "Line 1\n\nLine 2\n\nLine 3"
    assert strip_markdown(text) == expected
    assert count_words(text) == 6


def test_strip_markdown_contractions_and_punctuation():
    """Test that words with contractions or hyphens are counted properly."""
    text = "It's a self-explanatory word-count."
    assert count_words(text) == 4


def test_strip_markdown_snake_case():
    """Test that snake_case words (internal underscores) are not corrupted by emphasis stripping."""
    text = "This is a some_word_with_underscores and another_one."
    assert strip_markdown(text) == "This is a some_word_with_underscores and another_one."


def test_sentencize_basic():
    """Test basic sentence segmentation using sentencize."""
    text = "Tämä on ensimmäinen lause. Tämä on toinen lause! Ja kolmas?"
    expected = [
        "Tämä on ensimmäinen lause.",
        "Tämä on toinen lause!",
        "Ja kolmas?",
    ]
    assert sentencize(text) == expected


def test_sentencize_abbreviations():
    """Test that common abbreviations do not split sentences incorrectly if possible."""
    text = "Matti osti esim. omenoita. Pekka puolestaan osti päärynöitä."
    res = sentencize(text)
    assert len(res) == 2
    assert res[0].startswith("Matti")
    assert res[1].startswith("Pekka")


def test_is_markdown_detection():
    """Test that is_markdown correctly identifies markdown formats."""
    # Plain text should not be detected as markdown
    assert is_markdown("This is a simple plain text string.") is False
    assert is_markdown(None) is False
    assert is_markdown("") is False

    # Markdown indicators should be detected
    assert is_markdown("---\nlayout: post\n---\nHello") is True  # Front matter
    assert is_markdown("# Heading 1\nSome text") is True  # Header
    assert is_markdown("Here is a [link](http://example.com)") is True  # Link
    assert is_markdown("List item:\n* item 1") is True  # List item
    assert is_markdown("Code block:\n```python\nprint(1)\n```") is True  # Code block
    assert is_markdown("> This is a blockquote.") is True  # Blockquote
    assert is_markdown("This has **bold** text.") is True  # Bold
