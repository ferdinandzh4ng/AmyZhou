# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5001

# Set environment variable
ENV PORT=5001
ENV FLASK_ENV=production

# Run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001", "--workers", "2", "--threads", "2", "--timeout", "120"]

