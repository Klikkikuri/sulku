import os
from pprint import pprint

from ..utils import count_words

from ..dataset.reader import DatasetItem
from .models import ArticleSummary, StyleVector

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
import openai

INSTRUCTIONS_SUMMARY = """
You are a semantic decompression engine. 
Your task is to take the provided telegraphic text and extract the key information for reconstruction.

Rules:
    1. Do not add any new facts, names, or numbers that are not present in the input.
    2. Add only the necessary grammar, syntax, and connective tissue (prepositions, articles) to make it readable.
    3. Output ONLY the reconstructed structure.
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

Respond ONLY with the article text, in markdown format, without any additional commentary or explanation.

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
    type=StyleVector.model_fields["type"].description
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

{% if article.metadata['language'] == "fi" %}
Kieli: Kirjoitetun artikkelin **tulee** olla **suomeksi**. Artikkelin tulee olla sujuvaa, luonnollista ja ammattimaista suomen kieltä – kuin toimittajan kirjoittamaa.

Artikkelin aiheet:
{% for subject in article.metadata['subjects'] %}
- {{ subject }}
{% endfor %}
{% else %}
Language: The written article **must** be in **English**. The article should be fluent, natural, and professional English – as if written by a journalist.

Subjects of the article:
{% for subject in article.metadata['subjects'] %}
- {{ subject }}
{% endfor %}
{% endif %}
"""
""" Instructions to create synthetic article from summary and frontmatter """


def create_client():

    gemini = os.getenv("GEMINI_API_KEY")

    if gemini:
        client = openai.OpenAI(api_key=gemini, base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    else:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    return client

def prompt(template: str, **template_args: object) -> str:
    env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
    compiled = env.from_string(template)
    return compiled.render(**template_args).strip()

def summarize_text(text: str) -> ArticleSummary:
    client = create_client()
    response = client.chat.completions.parse(
        model="gemini-3.1-flash-lite",
        messages=[
            {"role": "system", "content": prompt(INSTRUCTIONS_SUMMARY)},
            {"role": "user", "content": prompt(ARTICLE_CONTEXT, text=text)},
        ],
        temperature=0,
        response_format=ArticleSummary
    )
    return response.choices[0].message.parsed  # type: ignore

def create_synthetic_article(article: DatasetItem, summary: ArticleSummary) -> str | None:
    """
    Create a synthetic article by combining the original article frontmatter and its summary.

    This function is useful for generating training data for machine learning models.
    """

    est_tokens = len(article.content) / 4  # Rough estimate: 1 token ~ 4 characters
    words = count_words(article.content)

    client = create_client()
    response = client.chat.completions.create(
        model="gemini-3.1-flash-lite",
        reasoning_effort="none",
        messages=[
            {"role": "system", "content": prompt(INSTRUCTIONS_JOURNALISM)},
            {"role": "user", "content": prompt(INSTRUCTIONS_SYNTHETIC, article=article, summary=summary, words=words)},
        ],
        temperature=0,
        max_tokens=int(est_tokens * 1.5),  # Allow for expansion in the generated article
    )

    pprint(response)

    return response.choices[0].message.content

