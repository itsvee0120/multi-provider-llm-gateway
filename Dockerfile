FROM python:3.11-slim

# uv pinned to the version that wrote uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock change.
# --no-install-project: the app imports src.* directly, so the project itself isn't installed.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
COPY .streamlit/ .streamlit/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8501

# API keys are not part of the image; docker-compose passes them at runtime from .env.
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
