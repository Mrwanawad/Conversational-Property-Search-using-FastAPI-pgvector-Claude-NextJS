FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install . \
    && useradd --create-home --uid 10001 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/api/health')" || exit 1
# Hosts like Render inject $PORT; the app seeds the synthetic inventory on first boot.
CMD ["sh", "-c", "exec uvicorn propsearch.api:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'"]
