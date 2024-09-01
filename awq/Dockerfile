# Use Python 3.11
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME=/srt-model-quantizing

# Set work directory
WORKDIR $APP_HOME

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . $APP_HOME/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -U -r requirements.txt

# Create directories for volume mounts
RUN mkdir -p $APP_HOME/data $APP_HOME/logs

# Set volume mount points
VOLUME ["$APP_HOME/data", "$APP_HOME/logs"]

# Run the application
CMD ["python", "app/main.py"]