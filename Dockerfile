FROM python:3.12-slim

# System deps for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    zlib1g-dev libjpeg-dev libpng-dev libfreetype6-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download fonts
RUN python download_fonts.py

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
