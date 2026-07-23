# Single-stage lightweight build - Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 1. Install CPU-only PyTorch FIRST from the official CPU wheel index
#    This must be done separately BEFORE other deps so pip resolves correctly
RUN pip install --no-cache-dir \
    torch==2.3.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Copy and install all other requirements (sentence-transformers, chromadb, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy application code (incl. pre-built data/bm25 and data/chroma indexes)
COPY . .

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
