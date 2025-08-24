# Lightweight CPU-only container for Streamlit + PyTorch
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# First try standard wheels; if torch misses, fallback to CPU-only index
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.2.0+cpu || true

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
