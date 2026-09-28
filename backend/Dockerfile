FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgomp1 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cài packages nhẹ trước (cache riêng, ít thay đổi)
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

# Cài packages nặng sau (cache riêng, hầu như không thay đổi)
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Copy code — layer này rebuild thường xuyên nhất
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
