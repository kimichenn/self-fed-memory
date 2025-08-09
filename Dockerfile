FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Copy entire repository (for simplicity/reliability)
COPY . /app

# Install project in editable mode with UI extras
RUN pip install --upgrade pip && \
    pip install --no-cache-dir ".[ui]"

# Default port for API
EXPOSE 8000

# Run as non-root for better security
USER 1000:1000

# Default command (override in docker-compose)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
