# Aria — container image for the Streamlit app.
# Stays lightweight by default (no torch): the deterministic hashing embedder works
# offline. For real semantic embeddings, install the embeddings extra (adds torch) and
# set ARIA_EMBEDDER=sentence_transformer.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARIA_DATA_DIR=/data \
    ARIA_PERSIST=true

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[ui,rag,viz]"

COPY app.py ./

# Run as a non-root user with a writable data volume.
RUN useradd --create-home aria && mkdir -p /data && chown -R aria /data /app
USER aria
VOLUME ["/data"]
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
