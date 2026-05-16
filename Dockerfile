# ── Indian Legal RAG — FastAPI Backend ──
# Deploy on Render as a Docker web service

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY app/ ./app/
COPY data/ ./data/

# Expose port (Render injects PORT env var)
EXPOSE 8000

# Start uvicorn — Render sets $PORT automatically
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
