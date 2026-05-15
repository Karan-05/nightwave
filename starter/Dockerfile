FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/starter

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY starter /app/starter
COPY case_data.json questions.json questions_adversarial.json nightwave.db /app/
COPY evidence /app/evidence
COPY schema /app/schema

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[api]"

EXPOSE 8000

CMD ["uvicorn", "nightwave.api:app", "--host", "0.0.0.0", "--port", "8000"]
