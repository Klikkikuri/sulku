"""
Tests for sulku_gradio application logic.
"""

from sulku_gradio.app import detect_input_language, format_model_choices


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
