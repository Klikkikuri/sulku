"""
Gradio UI Application Definition
================================

This module defines the Gradio interface that connects to the decoupled
Sulku HTTP API endpoint (`/api/v1/aidetect/`).
"""

import html
from typing import Any
import httpx
import gradio as gr


def format_paragraph_card(idx: int, p_text: str, p_score: float | None) -> str:
    """
    Format a single paragraph as a card with colored left border accent and score badge.

    :param idx: Paragraph 1-based index
    :param p_text: Raw paragraph text
    :param p_score: AI probability score (0.0 - 1.0) or None if excluded/short
    :return: HTML snippet for the paragraph card
    """
    if p_score is not None:
        score = max(0.0, min(1.0, float(p_score)))
        hue = int((1.0 - score) * 120)
        border_color = f"hsl({hue}, 75%, 45%)"
        bg_color = f"hsla({hue}, 70%, 50%, 0.07)"
        badge_bg = f"hsla({hue}, 75%, 45%, 0.15)"
        badge_text_color = f"hsl({hue}, 80%, 25%)"
        badge_text = f"Score: {score:.4f}"
    else:
        border_color = "#9ca3af"
        bg_color = "rgba(156, 163, 175, 0.07)"
        badge_bg = "rgba(156, 163, 175, 0.2)"
        badge_text_color = "#4b5563"
        badge_text = "Excluded / Short"

    escaped_text = html.escape(p_text)

    return (
        f'<div style="margin-bottom: 12px; padding: 12px 16px; border-left: 4px solid {border_color}; '
        f'background-color: {bg_color}; border-radius: 0 8px 8px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04);">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<span style="font-weight: 600; font-size: 0.9em; color: #4b5563;">Paragraph {idx}</span>'
        f'<span style="font-size: 0.825em; font-weight: 600; padding: 3px 10px; border-radius: 12px; '
        f'background-color: {badge_bg}; color: {badge_text_color};">{badge_text}</span>'
        f'</div>'
        f'<div style="font-size: 0.975em; line-height: 1.6; color: #1f2937; white-space: pre-wrap;">{escaped_text}</div>'
        f'</div>'
    )


def fetch_available_models(api_url: str) -> list[dict[str, Any]]:
    """Fetch available model info objects (name, loaded, languages) from the API server."""
    endpoint = f"{api_url.rstrip('/')}/api/v1/aidetect/models"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(endpoint)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("models", [])
    except Exception:
        pass
    return []


def format_model_choices(model_infos: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Format model choices with language code labels for Gradio CheckboxGroup."""
    choices = []
    for m in model_infos:
        name = m.get("name", "")
        langs = m.get("languages", [])
        lang_str = ", ".join(langs) if langs else "all"
        label = f"{name} ({lang_str})"
        choices.append((label, name))
    return choices


def detect_input_language(text: str) -> tuple[str, float]:
    """Detect ISO language code and confidence score from input text using fast-langdetect."""
    cleaned = text.strip()
    if not cleaned:
        return "", 0.0
    try:
        from fast_langdetect import detect

        res = detect(cleaned.replace("\n", " "))
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            return str(res[0].get("lang", "")), float(res[0].get("score", 0.0))
        elif isinstance(res, dict):
            return str(res.get("lang", "")), float(res.get("score", 0.0))
    except Exception:
        pass
    return "", 0.0


def analyze_input(
    text_or_url: str,
    api_url: str,
    selected_models: list[str],
    p_stay: float,
    alpha: float,
):
    """
    Sends text or URL content to the Sulku API endpoint and formats results for Gradio.
    """
    cleaned_input = text_or_url.strip()
    if not cleaned_input:
        return (
            '<div style="background: #f3f4f6; border: 1px solid #d1d5db; color: #4b5563; padding: 12px 16px; border-radius: 8px;">Please enter text or a URL to analyze.</div>',
            {},
            "",
            [],
        )

    try:
        from sulku.utils import prepare_input

        payload_content, content_type, display_name = prepare_input(cleaned_input)
    except Exception as exc:
        return (
            f'<div style="background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 12px 16px; border-radius: 8px;">❌ Pre-processing Error: Failed to fetch or extract input ({html.escape(str(exc))})</div>',
            {},
            "",
            [],
        )

    endpoint = f"{api_url.rstrip('/')}/api/v1/aidetect/"
    params: dict = {"p_stay": p_stay, "alpha": alpha}
    if selected_models:
        params["models"] = selected_models

    headers = {"Content-Type": f"{content_type}; charset=utf-8"}
    try:
        from opentelemetry.propagate import inject

        inject(headers)
    except Exception:
        pass

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                endpoint,
                content=payload_content.encode("utf-8"),
                headers=headers,
                params=params,
            )

        if resp.status_code != 200:
            err_detail = (
                resp.json().get("detail", resp.text)
                if resp.headers.get("content-type") == "application/json"
                else resp.text
            )
            return (
                f'<div style="background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 12px 16px; border-radius: 8px;">❌ API Error ({resp.status_code}): {html.escape(str(err_detail))}</div>',
                {},
                "",
                [],
            )

        data = resp.json()
    except Exception as exc:
        return (
            f'<div style="background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; padding: 12px 16px; border-radius: 8px;">❌ Connection Error: Failed to reach Sulku API at <code>{html.escape(endpoint)}</code> ({html.escape(str(exc))})</div>',
            {},
            "",
            [],
        )

    # Format Overview HTML
    is_ai = data.get("is_ai", False)
    verdict_emoji = "✨ AI-GENERATED" if is_ai else "👤 HUMAN-WRITTEN"
    verdict_color = "#dc2626" if is_ai else "#16a34a"
    badge_bg = "#fef2f2" if is_ai else "#f0fdf4"
    card_bg = "rgba(254, 242, 242, 0.5)" if is_ai else "rgba(240, 253, 244, 0.5)"
    card_border = "#fca5a5" if is_ai else "#86efac"

    final_score = data.get("final_score", 0.0)
    final_conf = data.get("final_confidence", 0.0)
    z_score = data.get("final_z_score", 0.0)
    ai_votes = data.get("ai_votes", 0)
    total_models = data.get("total_models", 0)

    summary_html = f"""
<div style="background: {card_bg}; border: 1px solid {card_border}; border-radius: 10px; padding: 16px; margin-bottom: 12px;">
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
        <h3 style="margin: 0; font-size: 1.05em; color: #374151;">Classification Verdict</h3>
        <span style="background: {badge_bg}; color: {verdict_color}; font-weight: 700; font-size: 0.875em; padding: 4px 12px; border-radius: 20px; border: 1px solid {card_border};">
            {verdict_emoji}
        </span>
    </div>

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 10px;">
        <div style="background: rgba(255,255,255,0.75); border: 1px solid rgba(200,200,200,0.35); border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7em; color: #6b7280; text-transform: uppercase; font-weight: 600;">Ensemble Score</div>
            <div style="font-size: 1.15em; font-weight: 700; color: #111827; margin-top: 2px;">{final_score:.4f}</div>
        </div>
        <div style="background: rgba(255,255,255,0.75); border: 1px solid rgba(200,200,200,0.35); border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7em; color: #6b7280; text-transform: uppercase; font-weight: 600;">Confidence</div>
            <div style="font-size: 1.15em; font-weight: 700; color: #111827; margin-top: 2px;">{final_conf:.1%}</div>
        </div>
        <div style="background: rgba(255,255,255,0.75); border: 1px solid rgba(200,200,200,0.35); border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7em; color: #6b7280; text-transform: uppercase; font-weight: 600;">Z-Score</div>
            <div style="font-size: 1.15em; font-weight: 700; color: #111827; margin-top: 2px;">{z_score:.2f}</div>
        </div>
        <div style="background: rgba(255,255,255,0.75); border: 1px solid rgba(200,200,200,0.35); border-radius: 8px; padding: 8px 10px; text-align: center;">
            <div style="font-size: 0.7em; color: #6b7280; text-transform: uppercase; font-weight: 600;">Model Consensus</div>
            <div style="font-size: 1.15em; font-weight: 700; color: #111827; margin-top: 2px;">{ai_votes} / {total_models}</div>
        </div>
    </div>
</div>
"""

    # Model scores table/dictionary for Label output
    predictions = data.get("predictions", {})

    # Paragraph breakdown table & display
    paragraphs = data.get("paragraphs", [])
    para_rows = []
    para_html_blocks = []

    for idx, p in enumerate(paragraphs, start=1):
        p_score = p.get("final_score")
        score_str = f"{p_score:.4f}" if p_score is not None else "Excluded/Short"
        p_text = p.get("text", "")
        sentences = p.get("sentences", [])

        para_rows.append([idx, score_str, p_text, len(sentences)])
        para_html_blocks.append(format_paragraph_card(idx, p_text, p_score))

    para_html = "".join(para_html_blocks)

    return summary_html, predictions, para_html, para_rows


def create_ui(default_api_url: str = "http://127.0.0.1:8000") -> gr.Blocks:
    """
    Constructs the Gradio Blocks UI layout with a 2-column input & results dashboard.
    """
    initial_model_infos = fetch_available_models(default_api_url)
    initial_choices = format_model_choices(initial_model_infos)
    initial_model_names = [m["name"] for m in initial_model_infos]

    custom_css = """
    .main-header { margin-bottom: 16px; }
    .input-panel { gap: 12px; }
    .results-panel { gap: 16px; }
    """

    with gr.Blocks(title="Sulku AI Detector UI", css=custom_css) as demo:
        gr.Markdown(
            """
            # 🌌 Sulku AI Text Detector
            *Visual testing interface for the Sulku AI detection HTTP API service.*
            """,
            elem_classes=["main-header"],
        )

        with gr.Row(equal_height=False):
            # Left Column: Input & Controls
            with gr.Column(scale=2, elem_classes=["input-panel"]):
                input_text = gr.Textbox(
                    label="Input Text or Article URL",
                    placeholder="Paste article text here or enter a URL (e.g. https://yle.fi/a/...)...",
                    lines=10,
                )

                with gr.Group():
                    with gr.Row(equal_height=True):
                        language_dropdown = gr.Dropdown(
                            choices=["Any", "fi", "en", "sv"],
                            value="Any",
                            label="Target Language",
                            info="Filter models by language support.",
                            scale=3,
                        )
                        detect_lang_btn = gr.Button("Detect Language", size="sm", scale=1)

                detected_lang_info = gr.Markdown("🌐 Language status: *Ready for input*")

                model_selector = gr.CheckboxGroup(
                    choices=initial_choices,
                    value=initial_model_names,
                    label="Active Models Ensemble",
                    info="All available models are shown. Models matching selected language are pre-selected.",
                )

                with gr.Accordion("Advanced Settings", open=False):
                    api_url_input = gr.Textbox(
                        label="Sulku API Base URL",
                        value=default_api_url,
                    )
                    refresh_models_btn = gr.Button("Refresh Available Models", size="sm")
                    p_stay_slider = gr.Slider(
                        minimum=0.5,
                        maximum=0.99,
                        value=0.85,
                        step=0.01,
                        label="HMM P(Stay) Transition Probability",
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.1,
                        maximum=5.0,
                        value=1.0,
                        step=0.1,
                        label="Alpha Smoothing Parameter",
                    )

                analyze_btn = gr.Button("Analyze Text", variant="primary")

            # Right Column: Results Dashboard
            with gr.Column(scale=3, elem_classes=["results-panel"]):
                verdict_html = gr.HTML()
                model_scores = gr.Label(label="Per-Model Confidence Scores")

                with gr.Tabs():
                    with gr.TabItem("Paragraph Scores Breakdown"):
                        paragraph_html = gr.HTML()
                    with gr.TabItem("Paragraph Data Table"):
                        paragraph_table = gr.Dataframe(
                            headers=["#", "Paragraph Score", "Text Snippet", "Sentence Count"],
                            datatype=["number", "str", "str", "number"],
                            interactive=False,
                        )

        def update_models_for_language(choice: str, api_url: str):
            model_infos = fetch_available_models(api_url)
            choices = format_model_choices(model_infos)
            all_names = [m["name"] for m in model_infos]

            if not choice or choice == "Any":
                return gr.update(choices=choices, value=all_names)

            target_lang = choice.lower().strip()
            matching_names = []
            for m in model_infos:
                m_langs = [lang_code.lower() for lang_code in m.get("languages", [])]
                if not m_langs or target_lang in m_langs:
                    matching_names.append(m["name"])

            return gr.update(choices=choices, value=matching_names)

        def on_detect_language_clicked(text: str, api_url: str):
            if not text.strip():
                return "Any", "🌐 Language status: *No text provided*", update_models_for_language("Any", api_url)

            detected_lang, score = detect_input_language(text)
            if detected_lang and score > 0.1:
                status_msg = f"🌐 Auto-detected Language: **{detected_lang.upper()}** (confidence: {score:.1%})"
                selected_lang = detected_lang
            else:
                status_msg = "🌐 Language status: *Could not reliably detect language*"
                selected_lang = "Any"

            model_update = update_models_for_language(selected_lang, api_url)
            return selected_lang, status_msg, model_update

        demo.load(
            fn=update_models_for_language,
            inputs=[language_dropdown, api_url_input],
            outputs=[model_selector],
        )

        detect_lang_btn.click(
            fn=on_detect_language_clicked,
            inputs=[input_text, api_url_input],
            outputs=[language_dropdown, detected_lang_info, model_selector],
        )

        language_dropdown.change(
            fn=update_models_for_language,
            inputs=[language_dropdown, api_url_input],
            outputs=[model_selector],
        )

        refresh_models_btn.click(
            fn=update_models_for_language,
            inputs=[language_dropdown, api_url_input],
            outputs=[model_selector],
        )

        analyze_btn.click(
            fn=analyze_input,
            inputs=[
                input_text,
                api_url_input,
                model_selector,
                p_stay_slider,
                alpha_slider,
            ],
            outputs=[verdict_html, model_scores, paragraph_html, paragraph_table],
        )

    return demo
