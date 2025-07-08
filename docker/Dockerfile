# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PDM
RUN pip install pdm

# Copy project files
COPY pyproject.toml pdm.lock* ./
COPY src/ ./src/
COPY gradio_app.py ./

# Install dependencies using PDM
RUN pdm install --prod --no-editable

# Create directories for mounted volumes
RUN mkdir -p /app/data /app/graphrag_output /app/config

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the Gradio app
CMD ["pdm", "run", "python", "gradio_app.py"]