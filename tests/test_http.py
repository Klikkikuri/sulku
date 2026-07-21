from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sulku.constants import LABEL_AI
from sulku.http import create_app


def test_health_check():
    """Verify that the health check endpoint returns 200 and healthy status."""
    client = TestClient(create_app())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_classify_text_success():
    """Test AI detection text classification endpoint with mocked fastText models."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.85])

    # Patch fasttext.load_model during TestClient context entry (lifespan)
    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/v1/aidetect/",
                content="This is a long test string that satisfies the minimum length requirement.",
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 200
            json_data = response.json()
            print("DEBUG: json_data is", json_data)
            print("DEBUG: mock_model.predict calls:", mock_model.predict.call_args_list)
            mock_model.predict.assert_called_once()
            assert json_data["is_ai"] is False
            assert json_data["ai_votes"] == 1
            assert json_data["final_score"] == 0.85
            assert "gemini-3.1-flash-lite" in json_data["predictions"]


def test_classify_text_validation_error():
    """Test validation errors for classify text (e.g. text too short)."""
    with patch("sulku.http.fasttext.load_model"):
        with TestClient(create_app()) as client:
            response = client.post(
                "/api/v1/aidetect/",
                content="short",
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 422


@patch("sulku.wpapi.router.get_wikipedia_page")
def test_wikipedia_page_success(mock_get_page):
    """Test Wikipedia page fetching endpoint with mocked service."""
    mock_get_page.return_value = {
        "title": "Python (programming language)",
        "page_id": 23862,
        "revision_id": 123456789,
        "timestamp": "2026-07-21T21:00:00Z",
        "editor": "TestEditor",
        "is_clean_evaluated": True,
        "editor_revert_rate": 0.05,
        "evaluation_notes": "Page version is clean.",
        "content": "Python is an interpreted programming language.",
    }

    client = TestClient(create_app())
    response = client.get("/api/v1/wikipedia/page?title=Python_(programming_language)")
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Python (programming language)"
    assert data["content"] == "Python is an interpreted programming language."


def test_classify_text_markdown_success():
    """Test AI detection text classification endpoint with markdown containing frontmatter."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.85])

    # Patch fasttext.load_model during TestClient context entry (lifespan)
    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            markdown_content = (
                "---\n"
                "title: Test Markdown\n"
                "author: Author\n"
                "---\n"
                "# Heading\n"
                "This is the actual **content** that will be analyzed."
            )
            response = client.post(
                "/api/v1/aidetect/",
                content=markdown_content,
                headers={"Content-Type": "text/markdown"},
            )
            assert response.status_code == 200
            # Ensure model.predict was called with normalized plain text (no markdown markers/newlines/frontmatter).
            expected_content = "Heading This is the actual content that will be analyzed."
            mock_model.predict.assert_called_once_with(expected_content, k=1)


def test_classify_text_raw_markdown_success():
    """Test AI detection text classification endpoint with raw markdown body."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.85])

    # Patch fasttext.load_model during TestClient context entry (lifespan)
    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            markdown_content = (
                "---\n"
                "title: Test Markdown\n"
                "---\n"
                "# Heading\n"
                "This is the actual **content** that will be analyzed."
            )
            # Test raw text body with Content-Type text/markdown
            response = client.post(
                "/api/v1/aidetect/",
                content=markdown_content,
                headers={"Content-Type": "text/markdown"},
            )
            assert response.status_code == 200
            expected_content = "Heading This is the actual content that will be analyzed."
            mock_model.predict.assert_called_once_with(expected_content, k=1)


def test_classify_text_multiline_paragraphs_scored_per_paragraph():
    """Test multiline text is scored paragraph-by-paragraph without newline predict errors."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.9])

    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            content = (
                "This is paragraph one with enough words to classify.\n"
                "Still the first paragraph.\n\n"
                "This is paragraph two and it is also long enough for inference."
            )
            response = client.post(
                "/api/v1/aidetect/",
                content=content,
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 200
            assert response.json()["final_score"] == 0.9
            assert mock_model.predict.call_count == 3
            for call in mock_model.predict.call_args_list:
                assert "\n" not in call.args[0]


def test_classify_text_weighted_paragraph_aggregation():
    """Test model score uses paragraph word-count weighting."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        ((LABEL_AI,), [0.1]),
        ((LABEL_AI,), [0.9]),
    ]

    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with patch(
            "sulku.http.sentencize",
            side_effect=[
                ["Short paragraph."],
                ["This paragraph has many words and should dominate weighted aggregation."],
            ],
        ):
            with TestClient(create_app()) as client:
                content = "Para one.\n\nPara two."
                response = client.post(
                    "/api/v1/aidetect/",
                    content=content,
                    headers={"Content-Type": "text/plain"},
                )

                assert response.status_code == 200
                final_score = response.json()["final_score"]
                # Expected weighted score: (0.1*2 + 0.9*10) / (2+10) = 0.766666...
                assert abs(final_score - 0.7666666667) < 1e-6


def test_classify_text_markdown_too_short():
    """Test AI detection text classification endpoint with markdown that becomes too short after stripping."""
    mock_model = MagicMock()
    # Patch fasttext.load_model
    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            markdown_content = "---\n" "title: A very long frontmatter title here\n" "---\n" "short"
            response = client.post(
                "/api/v1/aidetect/",
                content=markdown_content,
                headers={"Content-Type": "text/markdown"},
            )
            assert response.status_code == 422
            assert "too short" in response.json()["detail"]


def test_classify_text_uses_frontmatter_language_for_sentencize():
    """Test markdown frontmatter language is used as sentencize language."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.85])

    with patch("sulku.http.fasttext.load_model", return_value=mock_model):
        with patch("sulku.http.sentencize", return_value=["Sentence one."]) as mock_sentencize:
            with TestClient(create_app()) as client:
                markdown_content = (
                    "---\n"
                    "title: Test Markdown\n"
                    "language: en\n"
                    "---\n"
                    "This is the actual content that will be analyzed."
                )
                response = client.post(
                    "/api/v1/aidetect/",
                    content=markdown_content,
                    headers={"Content-Type": "text/markdown"},
                )
                assert response.status_code == 200
                assert mock_sentencize.call_count >= 1
                assert mock_sentencize.call_args.kwargs["lang"] == "en"


def test_classify_text_binary_rejected():
    """Test that binary content types (e.g. image/png) are rejected with 415."""
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/v1/aidetect/",
            content=b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...",
            headers={"Content-Type": "image/png"},
        )
        assert response.status_code == 415
        assert "binary" in response.json()["detail"].lower()
