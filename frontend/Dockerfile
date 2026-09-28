FROM python:3.11-slim

WORKDIR /app

# Install curl + unzip + Node.js 20 (Reflex cần unzip cho bun, Node cho Next.js)
RUN apt-get update && apt-get install -y curl unzip && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Cài Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source
COPY . .

# Init Reflex (download bun + node_modules, tạo .web)
RUN reflex init

EXPOSE 3000 8001

# 0.0.0.0 để container expose ra ngoài
CMD ["reflex", "run", "--backend-host", "0.0.0.0"]
