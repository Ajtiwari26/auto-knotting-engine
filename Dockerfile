FROM python:3.11-slim

# Install ffmpeg for audio decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for Docker cache
COPY requirements-render.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/

# Expose port
EXPOSE 5000

ENV PYTHONUNBUFFERED=1

# Start with gunicorn for production
CMD ["gunicorn", "src.server:app", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "1", "--threads", "2"]
