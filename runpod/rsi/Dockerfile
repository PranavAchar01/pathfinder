FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod==1.7.7 ultralytics==8.3.55 transformers==4.48.0 pillow numpy

COPY handler.py .

CMD ["python", "-u", "handler.py"]
