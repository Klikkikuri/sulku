ARG PYTHON_VERSION=3.12
ARG PYTHON_BASE_IMAGE=python:${PYTHON_VERSION}-bookworm

ARG VIRTUAL_ENV="/app/.venv"

ARG UV_VERSION="0.5.20"

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM ${PYTHON_BASE_IMAGE} AS build

LABEL org.opencontainers.image.authors="klikkikuri@protonmail.com" \
    org.opencontainers.image.source="https://github.com/Klikkikuri/sulku" \
    org.opencontainers.image.url="https://github.com/Klikkikuri"

ARG VIRTUAL_ENV

ENV UV_COMPILE_BYTECODE=1 \
    # Copy from the cache instead of linking since it's a mounted volume
    UV_LINK_MODE=copy

# Python settings
ENV PYTHONUNBUFFERED=1

# Virtual environment settings
ENV VIRTUAL_ENV=${VIRTUAL_ENV} \
    PATH="${VIRTUAL_ENV}/bin/:${PATH}"

# Disable telemetry
ENV HAYSTACK_TELEMETRY_ENABLED="False" \
    ANONYMIZED_TELEMETRY="False"

# More traceable shell
SHELL [ "/bin/bash", "-exo", "pipefail", "-c" ]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
        dumb-init gosu

VOLUME [ "${VIRTUAL_ENV}" ]

COPY --from=uv-bin /uv /uvx ${VIRTUAL_ENV}/bin/

WORKDIR /app

# Create a virtual environment
RUN uv venv --allow-existing --seed "${VIRTUAL_ENV}" && \
    echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

# Install dependencies (layer-cached before copying source)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages,target=packages \
    uv sync --frozen --no-install-project --no-dev --no-group training

# Install the application
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-group training --package sulku

# Production stage — slim image, non-root user, venv copied from build

FROM python:${PYTHON_VERSION}-slim AS production

ARG VIRTUAL_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=${VIRTUAL_ENV} \
    PATH="${VIRTUAL_ENV}/bin/:${PATH}"

# Disable telemetry
ENV HAYSTACK_TELEMETRY_ENABLED="False" \
    ANONYMIZED_TELEMETRY="False" \
    SENTRY_ENVIRONMENT="production"

# Copy virtual environment and application code from build stage
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}
COPY --from=build /app /app

# Create non-root user
RUN useradd -m -u 1000 sulku && chown -R sulku:sulku /app

USER sulku

CMD [ "sulku", "serve", "--host", "0.0.0.0", "--port", "8000" ]

# Development stage — devcontainers base image, dev deps, vscode user

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION} AS development

ARG VIRTUAL_ENV

WORKDIR /app

COPY --chown=vscode:vscode --from=build /app /app

COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy \
    SENTRY_ENVIRONMENT="development" \
    VIRTUAL_ENV=${VIRTUAL_ENV} \
    PATH="${VIRTUAL_ENV}/bin/:${PATH}"

# Disable telemetry
ENV HAYSTACK_TELEMETRY_ENABLED="False" \
    ANONYMIZED_TELEMETRY="False" \
    CODEGRAPH_TELEMETRY=0

RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

# Install dev dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --dev --package sulku && \
    chown -R vscode:vscode /app/.venv

USER vscode
