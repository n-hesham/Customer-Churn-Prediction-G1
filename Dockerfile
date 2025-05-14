# Dockerfile
FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install system dependencies if any (e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package \
#  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project content into the image
# This includes src, data (placeholders), models (placeholders), tests etc.
# .dvc files will be copied, but not the actual data if it's in DVC remote.
# For running in Docker, you'd typically dvc pull inside the container or mount volumes.
COPY . .

# Create directories that the application expects, if not already created by scripts
# (config.py already does this, but good for explicitness or if config isn't run first)
RUN mkdir -p data/raw data/processed models/saved_preprocessing data/predictions data/reports

# Make src a discoverable Python package path
ENV PYTHONPATH=/app

# Default command (e.g., to run the training pipeline)
# You might want to change this or override it when running the container.
# CMD ["python", "src/train_pipeline.py"]

# If you want to run the predict script by default:
# CMD ["python", "-m", "src.predict_script", "--input_data", "data/input_for_prediction.csv", "--output_preds", "data/predictions/output.csv"]

# Expose port if you plan to run a web service (e.g., MLflow UI, FastAPI/Flask app)
# EXPOSE 5000 # For MLflow UI or Flask app
# EXPOSE 8000 # For FastAPI app