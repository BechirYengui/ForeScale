# syntax=docker/dockerfile:1
# inference-api -- lightweight, non-root image (OpenShift-compatible).
FROM python:3.11-slim

# No .pyc, unbuffered logs for clean kubectl/docker logs.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install deps first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# --- Non-root user (OpenShift assigns an arbitrary high UID; make the app dir
#     group-writable and owned by root group so it still works there). ---
RUN useradd --uid 10001 --gid 0 --no-create-home --shell /usr/sbin/nologin appuser \
    && chown -R 10001:0 /app && chmod -R g=u /app
USER 10001:0

# Port > 1024 so no NET_BIND_SERVICE / root is required.
EXPOSE 8080

# Container-level liveness as a backstop to the k8s probe.
HEALTHCHECK --interval=10s --timeout=2s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz').status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
