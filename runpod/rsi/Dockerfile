FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# numpy<2: ultralytics 8.3.55 still calls np.trapz, removed in NumPy 2.0 (breaks validation).
RUN pip install --no-cache-dir \
    runpod==1.7.7 ultralytics==8.3.55 transformers==4.48.0 pillow "numpy<2" requests

COPY handler.py .

CMD ["python", "-u", "handler.py"]
