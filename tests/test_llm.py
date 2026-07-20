"""
Unit tests for LLM summarization and article generation
======================================================

Tests the create_synthetic_article function, specifically checking that
too short generated articles trigger a retry prompting the LLM for a longer version,
and verifying the accumulated token usage.
"""

from unittest.mock import MagicMock, patch
import pytest

from sulku.dataset.reader import DatasetItem
from sulku.summarize.llm import create_synthetic_article
from sulku.summarize.models import ArticleSummary, StyleVector


@pytest.fixture
def mock_article():
    """Create a mock DatasetItem."""
    article = MagicMock(spec=DatasetItem)
    # 50 words original content
    article.content = "Word " * 50
    article.metadata = {
        "language": "en",
        "subjects": ["testing"],
        "datePublished": "2026-07-19T12:00:00Z",
    }
    return article


@pytest.fixture
def mock_summary():
    """Create a mock ArticleSummary."""
    return ArticleSummary(
        headline="Test Headline",
        summary=["Detail 1", "Detail 2"],
        style=StyleVector(
            tone="Clinical / Detached",
            perspective="Third-Person Objective",
            angle="Macroscopic/Analytical",
            audience=["General / Mass Public"],
            type="Hard News / Breaking News",
        ),
    )


@patch("sulku.summarize.llm.create_client")
def test_create_synthetic_article_no_retry(mock_create_client, mock_article, mock_summary):
    """Test that if the generated article is long enough, no retry occurs."""
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # 45 words generated content (which is 90% of 50 words, so >= 80% threshold)
    mock_choice = MagicMock()
    mock_choice.message.content = "Word " * 45
    mock_choice.message.refusal = None
    mock_choice.finish_reason = "stop"

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    mock_client.chat.completions.create.return_value = mock_response

    metadata = {}
    content = create_synthetic_article(
        article=mock_article,
        summary=mock_summary,
        model="test-model",
        metadata_out=metadata,
        min_length_ratio=0.8,
    )

    assert content == "Word " * 45
    # chat.completions.create should be called exactly once
    assert mock_client.chat.completions.create.call_count == 1
    assert metadata["token_usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


@patch("sulku.summarize.llm.create_client")
def test_create_synthetic_article_with_retry(mock_create_client, mock_article, mock_summary):
    """Test that if the generated article is too short, we retry and accumulate token usage."""
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    # First call response: 10 words (too short, < 80% of 50)
    mock_choice_1 = MagicMock()
    mock_choice_1.message.content = "Word " * 10
    mock_choice_1.message.refusal = None
    mock_choice_1.finish_reason = "stop"
    mock_usage_1 = MagicMock()
    mock_usage_1.prompt_tokens = 100
    mock_usage_1.completion_tokens = 20
    mock_usage_1.total_tokens = 120
    mock_response_1 = MagicMock()
    mock_response_1.choices = [mock_choice_1]
    mock_response_1.usage = mock_usage_1

    # Second call response: 48 words (long enough)
    mock_choice_2 = MagicMock()
    mock_choice_2.message.content = "Word " * 48
    mock_choice_2.message.refusal = None
    mock_choice_2.finish_reason = "stop"
    mock_usage_2 = MagicMock()
    mock_usage_2.prompt_tokens = 150
    mock_usage_2.completion_tokens = 60
    mock_usage_2.total_tokens = 210
    mock_response_2 = MagicMock()
    mock_response_2.choices = [mock_choice_2]
    mock_response_2.usage = mock_usage_2

    # Configure side effect for successive calls
    mock_client.chat.completions.create.side_effect = [mock_response_1, mock_response_2]

    metadata = {}
    content = create_synthetic_article(
        article=mock_article,
        summary=mock_summary,
        model="test-model",
        metadata_out=metadata,
        min_length_ratio=0.8,
    )

    assert content == "Word " * 48
    # Should have called completions.create twice
    assert mock_client.chat.completions.create.call_count == 2
    # Verify token usage is accumulated
    assert metadata["token_usage"] == {
        "prompt_tokens": 250,
        "completion_tokens": 80,
        "total_tokens": 330,
    }


@patch("sulku.summarize.llm.create_client")
def test_create_synthetic_article_empty_original(mock_create_client, mock_article, mock_summary):
    """Test that if the original article content is empty, no retry is triggered."""
    mock_article.content = ""
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.refusal = None
    mock_choice.finish_reason = "stop"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    mock_client.chat.completions.create.return_value = mock_response

    metadata = {}
    content = create_synthetic_article(
        article=mock_article,
        summary=mock_summary,
        model="test-model",
        metadata_out=metadata,
    )

    assert content == ""
    assert mock_client.chat.completions.create.call_count == 1


def test_check_completion_choice_refusal():
    """Test that _check_completion_choice raises ValueError on refusal."""
    from sulku.summarize.llm import _check_completion_choice

    mock_choice = MagicMock()
    mock_choice.message.refusal = "I cannot write that."
    mock_choice.finish_reason = "stop"

    with pytest.raises(ValueError) as excinfo:
        _check_completion_choice(mock_choice)
    assert "rejected by the model: I cannot write that." in str(excinfo.value)


def test_check_completion_choice_content_filter():
    """Test that _check_completion_choice raises ValueError on content filter."""
    from sulku.summarize.llm import _check_completion_choice

    mock_choice = MagicMock()
    mock_choice.message.refusal = None
    mock_choice.finish_reason = "content_filter"

    with pytest.raises(ValueError) as excinfo:
        _check_completion_choice(mock_choice)
    assert "rejected due to content filtering" in str(excinfo.value)


def test_check_completion_choice_length(caplog):
    """Test that _check_completion_choice logs a warning when token limit is reached."""
    from sulku.summarize.llm import _check_completion_choice
    import logging

    mock_choice = MagicMock()
    mock_choice.message.refusal = None
    mock_choice.finish_reason = "length"

    with caplog.at_level(logging.WARNING):
        with pytest.raises(ValueError):
            _check_completion_choice(mock_choice)
    assert "LLM generation reached the token limit" in caplog.text


@patch("sulku.summarize.llm.create_client")
def test_create_synthetic_article_rejection(mock_create_client, mock_article, mock_summary):
    """Test that create_synthetic_article raises ValueError on model rejection."""
    mock_client = MagicMock()
    mock_create_client.return_value = mock_client

    mock_choice = MagicMock()
    mock_choice.message.refusal = "Safety violation"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError) as excinfo:
        create_synthetic_article(article=mock_article, summary=mock_summary)
    assert "rejected by the model: Safety violation" in str(excinfo.value)


def test_create_client_contextvars():
    """Test that create_client uses a context variable to cache and retrieve the client."""
    from sulku.summarize.llm import create_client
    import os
    import contextvars

    # Save original env and context var state
    orig_env = os.environ.get("GEMINI_API_KEY")
    os.environ["GEMINI_API_KEY"] = "dummy-key"

    # We run the test in a clean context to avoid affecting other tests
    def run_test():
        # First call creates the client
        client_1 = create_client()
        assert client_1 is not None

        # Second call returns the exact same client instance
        client_2 = create_client()
        assert client_1 is client_2

        # A new empty context will recreate the client
        def run_in_new_context():
            client_new = create_client()
            assert client_new is not client_1
            assert create_client() is client_new

        new_ctx = contextvars.Context()
        new_ctx.run(run_in_new_context)

    try:
        # Run inside a fresh context so we don't pollute the global/thread context
        test_ctx = contextvars.Context()
        test_ctx.run(run_test)
    finally:
        # Restore environment
        if orig_env is None:
            del os.environ["GEMINI_API_KEY"]
        else:
            os.environ["GEMINI_API_KEY"] = orig_env


