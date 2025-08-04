FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_irt.txt .
RUN pip install --no-cache-dir -r requirements_irt.txt

# Copy application files
COPY simple_backend.py .
COPY irt_personality_test.py .
COPY test_50_questions.py .
COPY data_persistence.py .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data && chmod 755 /app/data

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PORT=7860
ENV PYTHONPATH=/app
ENV STORAGE_TYPE=file
ENV HF_SPACE=true

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/test || exit 1

# Expose the port
EXPOSE 7860

# Run the application with proper production settings
CMD ["python", "-c", "import uvicorn; from simple_backend import app; uvicorn.run(app, host='0.0.0.0', port=7860, workers=1)"]
