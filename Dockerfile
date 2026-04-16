FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Python + system deps for OpenCV / Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    zlib1g-dev libjpeg-dev libpng-dev libfreetype6-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python deps (pulls CUDA-enabled torch via extra-index-url in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Download fonts
RUN python download_fonts.py

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

CMD bash -c 'if [ -n "$GOOGLE_APPLICATION_CREDENTIALS_JSON" ]; then echo "$GOOGLE_APPLICATION_CREDENTIALS_JSON" > /tmp/gcloud-key.json; export GOOGLE_APPLICATION_CREDENTIALS=/tmp/gcloud-key.json; fi && python app.py'
