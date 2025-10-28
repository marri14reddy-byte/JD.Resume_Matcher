# Fly.io ready Dockerfile for Streamlit + Chroma persistence
# Uses slim Python image; installs dependencies via requirements.txt

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

# System deps (add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first for better caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the app code
COPY . .

# Streamlit config via CLI flags (toml also provided)
EXPOSE 8080

CMD ["streamlit", "run", "resume_matcher_rag.py", "--server.address=0.0.0.0", "--server.port=8080", "--server.headless=true"]
