# # Use Python 3.11 slim image for smaller size
# FROM python:3.11-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements first for better caching
# COPY requirements.txt .

# # Install Python dependencies
# RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# # Copy application files
# COPY app.py .
# COPY systemPrompt.py .
# COPY dataset.pdf .

# # Create directory for ChromaDB persistence
# RUN mkdir -p pharma_db

# # Expose Gradio default port
# EXPOSE 7860

# # Set environment variables
# ENV PYTHONUNBUFFERED=1

# # Run the application
# CMD ["python", "app.py"]



# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for Python packages with native builds)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --timeout=1000 -r requirements.txt

# Copy entire project including src folder
COPY . .

# Expose Gradio port
EXPOSE 7860

# Clean logging
ENV PYTHONUNBUFFERED=1

# Run modular Gradio app
CMD ["python", "-m", "src.app"]