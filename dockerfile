# Use Python 3.10 as the base image
FROM python:3.12.9-slim

# Set working directory in the container
WORKDIR /app

# Install dependencies
# System dependencies may be needed for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create the temp_audio directory
RUN mkdir -p /app/temp_audio

# Expose port for the FastAPI application
EXPOSE 8082

# Set environment variables (these can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=8082

# Command to run the application
CMD ["python","-u", "main.py"]
