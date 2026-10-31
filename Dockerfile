FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Dependencies của ứng dụng SOFTS + Streamlit
COPY softs_pm25_deploy/requirements.txt /app/requirements-app.txt
RUN pip install --no-cache-dir -r /app/requirements-app.txt

# Dependencies của DINOv3
COPY external/dinov3/requirements-dev.txt /app/requirements-dinov3.txt
RUN pip install --no-cache-dir -r /app/requirements-dinov3.txt

# Copy mã nguồn, model, checkpoint và cấu hình vào image
COPY . .

EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "softs_pm25_deploy/app.py", "--server.address=0.0.0.0", "--server.port=8501"]