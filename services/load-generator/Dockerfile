# syntax=docker/dockerfile:1
# load-generator -- non-root image.
# IMPORTANT: build with the *repo root* as context so ./libs is available:
#   docker build -f services/load-generator/Dockerfile -t forescale/load-generator .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Shared core lib (traffic curve) first, then service deps.
COPY libs/ /app/libs/
RUN pip install --no-cache-dir /app/libs

COPY services/load-generator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/load-generator/app.py .

RUN useradd --uid 10001 --gid 0 --no-create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/results && chown -R 10001:0 /app && chmod -R g=u /app
USER 10001:0

CMD ["python", "app.py"]
