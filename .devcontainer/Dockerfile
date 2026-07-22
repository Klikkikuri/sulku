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

ENV UV_VERSION=${UV_VERSION} \
    UV_COMPILE_BYTECODE=1 \
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
        # Install dumb-init for preventing zombie process lingering
        dumb-init gosu

# Install UV
VOLUME [ "${VIRTUAL_ENV}" ]

COPY --from=uv-bin /uv /uvx ${VIRTUAL_ENV}/bin/

WORKDIR /app

# Create a virtual environment
RUN uv venv --allow-existing --seed "${VIRTUAL_ENV}" && \
echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev


# Install the application
COPY . /app

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package sulku

CMD [ "uv", "run", "sulku", "serve", "--host", "0.0.0.0", "--port", "8000" ]

# Development stage

FROM mcr.microsoft.com/devcontainers/python:${PYTHON_VERSION} AS development

ARG VIRTUAL_ENV

WORKDIR /app

COPY --chown=vscode:vscode --from=build /app /app

# Ensure `uv` is available in development stage regardless of virtualenv contents
COPY --from=uv-bin /uv /uvx /usr/local/bin/

ENV UV_LINK_MODE=copy

ENV SENTRY_ENVIRONMENT="development"

ENV VIRTUAL_ENV=$VIRTUAL_ENV \
    PATH="${VIRTUAL_ENV}/bin/:${PATH}" \
    XDG_CONFIG_HOME="/app/instance"
    
# Disable telemetry
ENV HAYSTACK_TELEMETRY_ENABLED="False" \
    ANONYMIZED_TELEMETRY="False" \
    CODEGRAPH_TELEMETRY=0

# Install nodejs and npm
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get --no-install-recommends install -y \
        nodejs npm

# Install codegraph CLI
RUN npm install -g @colbymchenry/codegraph

RUN echo "source ${VIRTUAL_ENV}/bin/activate" >> /etc/bash.bashrc

# Install development dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --dev --package sulku && \
    chown -R vscode:vscode /app/.venv

USER vscode
