

from typing import TypedDict


class TokenUsageDict(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class StyleVectorDict(TypedDict):
    tone: str
    perspective: str
    angle: str
    audience: list[str]
    type: str


class GenerationDetailsDict(TypedDict, total=False):
    model: str
    date: str
    headline: str
    summary: list[str]
    style: StyleVectorDict
    token_usage: TokenUsageDict


class SyntheticFrontMatter(TypedDict, total=False):
    id: str
    language: str
    title: str
    generation_details: GenerationDetailsDict


class AuthorDict(TypedDict, total=False):
    name: str
    organization: str


class FrontMatter(TypedDict, total=False):
    id: str
    language: str
    title: str
    url: str
    datePublished: str
    dateModified: str
    authors: list[AuthorDict]
    subjects: list[str]