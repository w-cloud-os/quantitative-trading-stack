# Python 3.12+ is required by nautilus_trader (the old 3.11 base image could not install it).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QUANT_STACK_CONFIG_DIR=/app/configs

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Install dependencies from metadata first so this layer is cached between code edits.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
ARG EXTRAS=""
RUN uv pip install --system ".${EXTRAS}"

COPY configs/ ./configs/

# Run as an unprivileged user; the cache directory must be writable by it.
RUN useradd --create-home app && mkdir -p /app/.data_cache && chown -R app:app /app/.data_cache
USER app

EXPOSE 8501
CMD ["quant-stack", "dashboard", "--host", "0.0.0.0"]
