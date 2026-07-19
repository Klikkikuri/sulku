import os
import logging
from typing import Any, Optional

from ..utils import count_words

from ..dataset.reader import DatasetItem
from .models import ArticleSummary, StyleVector
from sulku.constants import DEFAULT_MODEL

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
import openai
from openai.types.chat import ChatCompletionMessageParam

logger = logging.getLogger(__name__)

INSTRUCTIONS_SUMMARY = """
You are a semantic decompression engine. 
Your task is to take the provided telegraphic text and extract the key information for reconstruction.

Rules:
    1. Do not add any new facts, names, or numbers that are not present in the input.
    2. Add only the necessary grammar, syntax, and connective tissue (prepositions, articles) to make it readable.
    3. Collect 5 to 10 key details and summarize them in a concise manner.
    4. Output ONLY the reconstructed structure.

"""

ARTICLE_CONTEXT = r"""
Extract from the following article:
<article>
```markdown
{{ text|indent(4) }}
```
</article>
"""

INSTRUCTIONS_JOURNALISM = """
You are a journalist tasked with writing an article based on the provided summary and details.

Respond ONLY with the article text, in markdown format, without any additional commentary or explanation. Use fully formated markdown, including headings, lists, and other elements as appropriate. Do not include any metadata or frontmatter in the output.

Use the following style guide to inform your writing:

## Style guide

The style axis dictates the mechanical and linguistic construction of an article, focusing on elements like vocabulary complexity, sentence length, and syntax. Writers calibrate this axis to control the piece's readability and aesthetic rhythm, ensuring the language practically supports the chosen tone and aligns with the target audience's comprehension level.

### Tone
{tone}
### Perspective
{perspective}
### Angle
{angle}
### Audience
{audience}
### Type
{type}
""".format(
    tone=StyleVector.model_fields["tone"].description,
    perspective=StyleVector.model_fields["perspective"].description,
    angle=StyleVector.model_fields["angle"].description,
    audience=StyleVector.model_fields["audience"].description,
    type=StyleVector.model_fields["type"].description,
)
""" Instructions to write article from summary and frontmatter """

INSTRUCTIONS_SYNTHETIC = """
Write the article in the style provided, about the event described in the key details.

Key details for the article:
{% for item in summary.summary %}
- {{ item }}
{% endfor %}

Try to keep the length at {{ words }} words, but it can be longer or shorter if necessary.

Article writing style:
 - **Tone**: {{ summary.style.tone }}
 - **Perspective**: {{ summary.style.perspective }}
 - **Angle**: {{ summary.style.angle }}
 - **Audience**: {{ summary.style.audience | join(", ") }}
 - **Type**: {{ summary.style.type }}

Published date: {{ article.metadata['datePublished'] }} – assume knowledge of events up to this date, but not beyond.

{% if article.metadata['language'] == "fi" %}
Kieli: Kirjoitetun artikkelin **tulee** olla **suomeksi**. Artikkelin tulee olla sujuvaa, luonnollista ja ammattimaista suomen kieltä – kuin toimittajan kirjoittamaa.

{% if article.metadata['subjects'] %}
Kun kirjoitat artikkelia, ota huomioon seuraavat aiheet, ja yritä sisällyttää ne artikkeliin, jos mahdollista. Ne voidaan ilmaista implisiittisesti tai eksplisiittisesti:
{% for subject in article.metadata['subjects'] %}
- {{ subject }}
{% endfor %}
{% endif %}

{% else %}

Language: The written article **must** be in **English**. The article should be fluent, natural, and professional English – as if written by a journalist.

{% if article.metadata['subjects'] %}
When writing the article, consider the following subjects and try to incorporate them into the article if possible. They can be implied or explicitly mentioned:
{% for subject in article.metadata['subjects'] %}
- {{ subject }}
{% endfor %}
{% endif %}
{% endif %}
"""
""" Instructions to create synthetic article from summary and frontmatter """

INSTRUCTIONS_RETRY_LONGER = """
The generated article is too short ({{ gen_words }} words). The article should be {{ words }} words. Please rewrite the article to be longer and more detailed, aiming for approximately {{ words }} words. Otherwise, follow the same style and content guidelines as before.
"""
""" Instructions to rewrite synthetic article to be longer """


def create_client():

    gemini = os.getenv("GEMINI_API_KEY")

    if gemini:
        client = openai.OpenAI(
            api_key=gemini,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        )
    else:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    return client


def prompt(template: str, **template_args: object) -> str:
    env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    compiled = env.from_string(template)
    return compiled.render(**template_args).strip()


def _check_completion_choice(choice: Any) -> None:
    """
    Check the completion choice for rejection or token limit issues.

    :param choice: The choice object from the API response.
    :raises ValueError: If the choice contains a refusal, content filter block, or hit token limits.
    """
    if getattr(choice.message, "refusal", None):
        raise ValueError(f"LLM request was rejected by the model: {choice.message.refusal}")
    if getattr(choice, "finish_reason", None) == "content_filter":
        raise ValueError("LLM request was rejected due to content filtering.")
    if getattr(choice, "finish_reason", None) == "length":
        # raise ValueError("LLM generation reached the token limit (finish_reason: length).")
        logger.warning("LLM generation reached the token limit (finish_reason: length).", extra={"choice": choice})


def summarize_text(text: str, model: str = DEFAULT_MODEL) -> ArticleSummary:
    client = create_client()
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt(INSTRUCTIONS_SUMMARY)},
            {"role": "user", "content": prompt(ARTICLE_CONTEXT, text=text)},
        ],
        temperature=0,
        response_format=ArticleSummary,
    )
    if not response.choices:
        raise ValueError("LLM response did not contain any choices.")
    _check_completion_choice(response.choices[0])
    parsed = response.choices[0].message.parsed
    usage = response.usage
    token_usage = (
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage
        else None
    )

    logger.info(
        "Summarized text using model %s",
        model,
        extra={
            "token_usage": token_usage,
            "generated_structure": parsed.model_dump() if parsed else None,
        },
    )

    return parsed  # type: ignore


def create_synthetic_article(
    article: DatasetItem,
    summary: ArticleSummary,
    model: str = DEFAULT_MODEL,
    metadata_out: Optional[dict[str, Any]] = None,
    min_length_ratio: float = 0.7,
) -> str | None:
    """
    Create a synthetic article by combining the original article frontmatter and its summary.

    This function is useful for generating training data for machine learning models.

    If the generated article is significantly shorter than the estimated original,
    the LLM is prompted to write a longer version.

    :param article: The original dataset item.
    :param summary: The generated article summary.
    :param model: The LLM model to use.
    :param metadata_out: Optional dictionary to collect generation metadata (e.g. token_usage).
    :param min_length_ratio: The minimum ratio of generated word count to estimated original word count.
                             If the generated article is shorter than this ratio, prompt the LLM to write
                             a longer version.
    :type min_length_ratio: float, optional
    """

    est_tokens = len(article.content) / 4  # Rough estimate: 1 token ~ 4 characters
    words = count_words(article.content)

    client = create_client()
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": prompt(INSTRUCTIONS_JOURNALISM)},
        {
            "role": "user",
            "content": prompt(
                INSTRUCTIONS_SYNTHETIC,
                article=article,
                summary=summary,
                words=words,
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        reasoning_effort="none",
        messages=messages,
        temperature=0,
        max_tokens=int(
            est_tokens * 1.5
        ),  # Allow for expansion in the generated article
    )

    if not response.choices:
        raise ValueError("LLM response did not contain any choices.")
    _check_completion_choice(response.choices[0])
    content = response.choices[0].message.content
    usage = response.usage
    token_usage = (
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage
        else None
    )

    gen_words = count_words(content)
    if content and words > 0 and gen_words < min_length_ratio * words:
        logger.info(
            "Generated synthetic article was too short (%d words, expected >= %d words). "
            "Prompting LLM for longer version.",
            gen_words,
            int(min_length_ratio * words),
        )
        messages.append({"role": "assistant", "content": content})
        messages.append({
            "role": "user",
            "content": prompt(
                INSTRUCTIONS_RETRY_LONGER,
                gen_words=gen_words,
                words=words,
            ),
        })

        response_retry = client.chat.completions.create(
            model=model,
            reasoning_effort="none",
            messages=messages,
            temperature=0,
            max_tokens=int(est_tokens * 1.5),
        )
        if not response_retry.choices:
            raise ValueError("LLM retry response did not contain any choices.")
        _check_completion_choice(response_retry.choices[0])
        content = response_retry.choices[0].message.content
        usage_retry = response_retry.usage
        if usage_retry:
            if token_usage is None:
                token_usage = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                }
            token_usage["prompt_tokens"] += usage_retry.prompt_tokens or 0
            token_usage["completion_tokens"] += usage_retry.completion_tokens or 0
            token_usage["total_tokens"] += usage_retry.total_tokens or 0

    logger.info(
        "Generated synthetic article using model %s",
        model,
        extra={
            "token_usage": token_usage,
            "generated_content": content,
        },
    )

    if metadata_out is not None:
        metadata_out["token_usage"] = token_usage

    return content
