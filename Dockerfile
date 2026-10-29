FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV & PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python ML & Cryptography packages
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir pycryptodome pyyaml scikit-learn matplotlib pandas timm albumentations

# Copy full application source code
COPY . .

ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
