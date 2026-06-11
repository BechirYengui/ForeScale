# syntax=docker/dockerfile:1
# forescale-controller -- non-root image.
# IMPORTANT: build with the *repo root* as context (needs ./libs and ./forecaster):
#   docker build -f services/forescale-controller/Dockerfile -t forescale/controller .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Shared core lib + forecaster package.
COPY libs/ /app/libs/
RUN pip install --no-cache-dir /app/libs

COPY services/forescale-controller/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The forecaster package is imported directly (kept on PYTHONPATH).
COPY forecaster/ /app/forecaster/
COPY services/forescale-controller/app.py .

RUN useradd --uid 10001 --gid 0 --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /models && chown -R 10001:0 /app /models && chmod -R g=u /app /models
USER 10001:0

ENV PYTHONPATH=/app

CMD ["python", "app.py"]
