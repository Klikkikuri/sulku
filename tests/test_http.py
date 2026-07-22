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
    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
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
    with patch("sulku.prediction.fasttext.load_model"):
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
    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
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
            mock_model.predict.assert_called_once_with(expected_content, k=2)


def test_classify_text_raw_markdown_success():
    """Test AI detection text classification endpoint with raw markdown body."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.85])

    # Patch fasttext.load_model during TestClient context entry (lifespan)
    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
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
            mock_model.predict.assert_called_once_with(expected_content, k=2)


def test_classify_text_multiline_paragraphs_scored_per_paragraph():
    """Test multiline text is scored paragraph-by-paragraph without newline predict errors."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.9])

    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            content = (
                "This is paragraph one with enough words to classify.\n"
                "Still the first paragraph of the document.\n\n"
                "This is paragraph two and it is also long enough for inference."
            )
            response = client.post(
                "/api/v1/aidetect/",
                content=content,
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 200
            # Under HMM smoothing, the scores are smoothed and reinforce each other.
            # Paragraph 1 has score 0.969613 (16 words), Paragraph 2 has score 0.9 (12 words).
            # Weighted average: (0.9696132596685083 * 16 + 0.9 * 12) / 28 = 0.9397790055248619
            assert abs(response.json()["final_score"] - 0.9397790055248619) < 1e-6

            # With p_stay=0.5, HMM smoothing is a no-op and we get the raw 0.9 score
            response_unsmoothed = client.post(
                "/api/v1/aidetect/?p_stay=0.5",
                content=content,
                headers={"Content-Type": "text/plain"},
            )
            assert response_unsmoothed.status_code == 200
            assert abs(response_unsmoothed.json()["final_score"] - 0.9) < 1e-6

            assert mock_model.predict.call_count == 6
            for call in mock_model.predict.call_args_list:
                assert "\n" not in call.args[0]


def test_classify_text_weighted_paragraph_aggregation():
    """Test model score uses paragraph word-count weighting."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        ((LABEL_AI,), [0.1]),
        ((LABEL_AI,), [0.9]),
    ]

    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        # Patch DEFAULT_LONG_PARAGRAPH_WORDS to 1 so the short paragraph (2 words) isn't filtered out
        with patch("sulku.prediction.DEFAULT_LONG_PARAGRAPH_WORDS", 1):
            with patch(
                "sulku.utils.sentencize",
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
    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            markdown_content = "---\ntitle: A very long frontmatter title here\n---\nshort"
            response = client.post(
                "/api/v1/aidetect/",
                content=markdown_content,
                headers={"Content-Type": "text/markdown"},
            )
            assert response.status_code == 422
            assert "too short" in response.json()["detail"]


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


def test_classify_text_weighted_sentence_aggregation_smoothed():
    """Test model score uses sentence-level word-count weighting with default HMM smoothing."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        ((LABEL_AI,), [0.1]),
        ((LABEL_AI,), [0.9]),
    ]

    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        with patch(
            "sulku.utils.sentencize",
            side_effect=[
                [
                    "Thank you!",
                    "This is a much longer sentence that has exactly ten words.",
                ],
            ],
        ):
            with TestClient(create_app()) as client:
                content = "Thank you! This is a much longer sentence that has exactly ten words."
                response = client.post(
                    "/api/v1/aidetect/",
                    content=content,
                    headers={"Content-Type": "text/plain"},
                )

                assert response.status_code == 200
                final_score = response.json()["final_score"]
                # Under HMM smoothing, the scores are smoothed and move towards each other:
                # 0.1 -> ~0.2826, 0.9 -> ~0.7174, weighted average = 0.65050167...
                assert abs(final_score - 0.6505016722408027) < 1e-6


def test_classify_text_weighted_sentence_aggregation_unsmoothed():
    """Test model score uses sentence-level word-count weighting without HMM smoothing (p_stay=0.5)."""
    mock_model = MagicMock()
    mock_model.predict.side_effect = [
        ((LABEL_AI,), [0.1]),
        ((LABEL_AI,), [0.9]),
    ]

    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        with patch(
            "sulku.utils.sentencize",
            side_effect=[
                [
                    "Thank you!",
                    "This is a much longer sentence that has exactly ten words.",
                ],
            ],
        ):
            with TestClient(create_app()) as client:
                content = "Thank you! This is a much longer sentence that has exactly ten words."
                response = client.post(
                    "/api/v1/aidetect/?p_stay=0.5",
                    content=content,
                    headers={"Content-Type": "text/plain"},
                )
                assert response.status_code == 200
                assert abs(response.json()["final_score"] - 0.7769230769) < 1e-6


def test_forward_backward_smoothing():
    """Test Forward-Backward HMM smoothing direct unit test."""
    from sulku.prediction import forward_backward_smoothing

    # Empty inputs
    assert forward_backward_smoothing([], []) == ([], [])

    # Single item sequence
    p_ai, p_hum = forward_backward_smoothing([0.8], [0.2])
    assert len(p_ai) == 1
    assert abs(p_ai[0] - 0.8) < 1e-6
    assert abs(p_hum[0] - 0.2) < 1e-6

    # Symmetric behavior (p_stay = 0.5 is no-op)
    raw_ai = [0.1, 0.9, 0.4]
    raw_hum = [0.9, 0.1, 0.6]
    smoothed_ai, smoothed_hum = forward_backward_smoothing(raw_ai, raw_hum, p_stay=0.5)
    for r_ai, s_ai in zip(raw_ai, smoothed_ai):
        assert abs(r_ai - s_ai) < 1e-6
    for r_hum, s_hum in zip(raw_hum, smoothed_hum):
        assert abs(r_hum - s_hum) < 1e-6

    # Normalization property: probabilities must sum to 1.0
    smoothed_ai, smoothed_hum = forward_backward_smoothing(raw_ai, raw_hum, p_stay=0.8, alpha=1.2)
    for ai, hum in zip(smoothed_ai, smoothed_hum):
        assert abs(ai + hum - 1.0) < 1e-6

    # Influence of neighbors: a low AI score surrounded by high AI scores should pull up
    raw_ai = [0.9, 0.1, 0.9]
    raw_hum = [0.1, 0.9, 0.1]
    smoothed_ai, _ = forward_backward_smoothing(raw_ai, raw_hum, p_stay=0.85, alpha=1.0)
    assert smoothed_ai[1] > 0.1  # pulled up by neighbors


def test_classify_text_per_paragraph_details():
    """Verify that per-paragraph details are returned correctly including predictions and final_score."""
    mock_model = MagicMock()
    mock_model.predict.return_value = ((LABEL_AI,), [0.8])

    with patch("sulku.prediction.fasttext.load_model", return_value=mock_model):
        with TestClient(create_app()) as client:
            content = (
                "This is the first sentence of paragraph one.\n"
                "This is the second sentence of paragraph one.\n\n"
                "Single sentence paragraph two.\n\n"
                "This is a very long single sentence paragraph that has more than "
                "fifteen words in it to test that it is not excluded."
            )
            response = client.post(
                "/api/v1/aidetect/?p_stay=0.5",
                content=content,
                headers={"Content-Type": "text/plain"},
            )
            assert response.status_code == 200
            json_data = response.json()

            assert "paragraphs" in json_data
            paragraphs = json_data["paragraphs"]
            assert len(paragraphs) == 3

            # Paragraph 1 should be classified (has > 1 sentences)
            assert paragraphs[0]["text"] == (
                "This is the first sentence of paragraph one.\n" "This is the second sentence of paragraph one."
            )
            assert paragraphs[0]["sentences"] == [
                "This is the first sentence of paragraph one.",
                "This is the second sentence of paragraph one.",
            ]
            assert paragraphs[0]["predictions"] is not None
            assert "gemini-3.1-flash-lite" in paragraphs[0]["predictions"]
            assert abs(paragraphs[0]["predictions"]["gemini-3.1-flash-lite"] - 0.8) < 1e-6
            assert abs(paragraphs[0]["final_score"] - 0.8) < 1e-6

            # Paragraph 2 should be excluded (only has 1 short sentence while Paragraph 1 has 2)
            assert paragraphs[1]["text"] == "Single sentence paragraph two."
            assert paragraphs[1]["sentences"] == ["Single sentence paragraph two."]
            assert paragraphs[1]["predictions"] is None
            assert paragraphs[1]["final_score"] is None

            # Paragraph 3 should be classified (has 1 long sentence >= 15 words)
            assert paragraphs[2]["text"] == (
                "This is a very long single sentence paragraph that has more than "
                "fifteen words in it to test that it is not excluded."
            )
            assert paragraphs[2]["sentences"] == [
                "This is a very long single sentence paragraph that has more than "
                "fifteen words in it to test that it is not excluded."
            ]
            assert paragraphs[2]["predictions"] is not None
            assert "gemini-3.1-flash-lite" in paragraphs[2]["predictions"]
            assert abs(paragraphs[2]["predictions"]["gemini-3.1-flash-lite"] - 0.8) < 1e-6
            assert abs(paragraphs[2]["final_score"] - 0.8) < 1e-6
