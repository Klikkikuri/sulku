"""
Tests for sulku_gradio application logic.
"""

from sulku_gradio.app import detect_input_language, format_model_choices, format_paragraph_card


def test_detect_input_language():
    lang_fi, score_fi = detect_input_language("Tämä on suomenkielistä tekstiä ja lausetta.")
    assert lang_fi == "fi"
    assert score_fi > 0.5

    lang_en, score_en = detect_input_language("This is a long English text sentence for language detection.")
    assert lang_en == "en"
    assert score_en > 0.5


def test_detect_input_language_empty():
    lang, score = detect_input_language("")
    assert lang == ""
    assert score == 0.0


def test_format_model_choices():
    models = [
        {"name": "m1", "languages": ["fi"]},
        {"name": "m2", "languages": ["en", "fi"]},
        {"name": "m3", "languages": []},
    ]
    choices = format_model_choices(models)
    assert choices == [
        ("m1 (fi)", "m1"),
        ("m2 (en, fi)", "m2"),
        ("m3 (all)", "m3"),
    ]


def test_format_paragraph_card_with_score():
    html_out = format_paragraph_card(1, "Test paragraph text", 0.85)
    assert "Paragraph 1" in html_out
    assert "Score: 0.8500" in html_out
    assert "border-left: 4px solid" in html_out
    assert "Test paragraph text" in html_out


def test_format_paragraph_card_excluded():
    html_out = format_paragraph_card(2, "Short text", None)
    assert "Paragraph 2" in html_out
    assert "Excluded / Short" in html_out


def test_format_paragraph_card_escapes_html():
    html_out = format_paragraph_card(3, "Text with <script>alert(1)</script>", 0.1)
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out

