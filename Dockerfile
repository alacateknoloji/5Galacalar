# 1) Official CUDA base image required by the competition.
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2) System packages. libgl1 + libglib2.0-0 are added on top of the doc's list
#    to avoid the common "libGL.so.1 not found" OpenCV import error.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3) Directory layout (input/output mounted by the judge at run time).
RUN mkdir -p /app/data/input /app/data/output /app/models /app/src

# 4) Dependencies. Build-time internet is available; runtime is offline.
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
# GPU build of PyTorch for CUDA 12.1 (matches the Tesla T4 VM).
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# 4b) IMPORTANT: EasyOCR downloads its detection/recognition models on first use.
#     Runtime is offline, so we pre-download them now (build-time internet is on)
#     and bake them into the image (~/.EasyOCR/). If you switch to PaddleOCR,
#     replace this with the equivalent PaddleOCR warm-up call.
RUN python3 -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# 5) Model weights. Place your .pt files in ./models before building.
COPY models/ /app/models/

# 6) Source + entry point (selective COPY to keep the image small).
COPY src/ /app/src/
COPY main.py .
COPY README.md .

# 7) Auto-start on `docker run` (no manual step).
CMD ["python3", "main.py"]
