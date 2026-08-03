"""
Gradio UI Application Definition
================================

This module defines the Gradio interface that connects to the decoupled
Sulku HTTP API endpoint (`/api/v1/aidetect/`).
"""

import httpx
import gradio as gr


def fetch_available_models(api_url: str) -> list[str]:
    """Fetch available model names from the API server."""
    endpoint = f"{api_url.rstrip('/')}/api/v1/aidetect/models"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(endpoint)
            if resp.status_code == 200:
                data = resp.json()
                return [m["name"] for m in data.get("models", []) if m.get("loaded", True)]
    except Exception:
        pass
    return []


def analyze_input(text_or_url: str, api_url: str, selected_models: list[str], p_stay: float, alpha: float):
    """
    Sends text or URL content to the Sulku API endpoint and formats results for Gradio.
    """
    cleaned_input = text_or_url.strip()
    if not cleaned_input:
        return "Please enter text or a URL to analyze.", {}, "", []

    try:
        from sulku.utils import prepare_input

        payload_content, content_type, display_name = prepare_input(cleaned_input)
    except Exception as exc:
        return f"❌ Pre-processing Error: Failed to fetch or extract input ({exc})", {}, "", []

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
            return f"❌ API Error ({resp.status_code}): {err_detail}", {}, "", []

        data = resp.json()
    except Exception as exc:
        return f"❌ Connection Error: Failed to reach Sulku API at `{endpoint}` ({exc})", {}, "", []

    # Format Overview Markdown
    is_ai = data.get("is_ai", False)
    verdict_emoji = "🤖 AI-GENERATED DETECTED" if is_ai else "👤 HUMAN-WRITTEN DETECTED"
    verdict_color = "red" if is_ai else "green"

    final_score = data.get("final_score", 0.0)
    final_conf = data.get("final_confidence", 0.0)
    z_score = data.get("final_z_score", 0.0)
    ai_votes = data.get("ai_votes", 0)
    total_models = data.get("total_models", 0)

    summary_md = f"""
### Verdict: <span style="color: {verdict_color};">{verdict_emoji}</span>

- **Ensemble Score**: `{final_score:.4f}` (Higher value indicates higher probability of AI generation)
- **Confidence**: `{final_conf:.2%}`
- **Stouffer Combined Z-Score**: `{z_score:.2f}`
- **Model Consensus**: `{ai_votes} / {total_models}` models voted AI
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

        badge = f"<b style='color: {'red' if p_score and p_score >= 0.5 else 'green'};'>[Score: {score_str}]</b>"
        para_html_blocks.append(f"<div><p><b>Paragraph {idx}</b> {badge}</p><blockquote style='background: #f9f9f9; padding: 8px;'>{p_text}</blockquote></div>")

    para_html = "".join(para_html_blocks)

    return summary_md, predictions, para_html, para_rows


def create_ui(default_api_url: str = "http://127.0.0.1:8000") -> gr.Blocks:
    """
    Constructs the Gradio Blocks UI layout.
    """
    initial_models = fetch_available_models(default_api_url)

    with gr.Blocks(title="Sulku AI Detector UI") as demo:
        gr.Markdown(
            """
            # 🌌 Sulku AI Text Detector
            *Visual testing interface for the Sulku AI detection HTTP API service.*
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input Text or Article URL",
                    placeholder="Paste article text here or enter a URL (e.g. https://yle.fi/a/...)...",
                    lines=10,
                )
                
                model_selector = gr.CheckboxGroup(
                    choices=initial_models,
                    value=initial_models,
                    label="Active Models Ensemble",
                    info="Select which fastText models to use for classification. Leave empty to use all.",
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

            with gr.Column(scale=3):
                verdict_md = gr.Markdown(label="Classification Verdict")
                model_scores = gr.Label(label="Per-Model Scores")

        with gr.Tabs():
            with gr.TabItem("Paragraph Scores Breakdown"):
                paragraph_html = gr.HTML()
            with gr.TabItem("Paragraph Data Table"):
                paragraph_table = gr.Dataframe(
                    headers=["#", "Paragraph Score", "Text Snippet", "Sentence Count"],
                    datatype=["number", "str", "str", "number"],
                    interactive=False,
                )

        def update_models(api_url: str):
            models = fetch_available_models(api_url)
            return gr.update(choices=models, value=models)

        refresh_models_btn.click(
            fn=update_models,
            inputs=[api_url_input],
            outputs=[model_selector],
        )

        analyze_btn.click(
            fn=analyze_input,
            inputs=[input_text, api_url_input, model_selector, p_stay_slider, alpha_slider],
            outputs=[verdict_md, model_scores, paragraph_html, paragraph_table],
        )

    return demo
