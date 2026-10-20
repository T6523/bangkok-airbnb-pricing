FROM python:3.12-slim

# libgomp1: OpenMP runtime needed by LightGBM and XGBoost; make: runs the pipeline
RUN apt-get update && apt-get install -y --no-install-recommends make libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# no .venv in the image, so point the Makefile at the system python
CMD ["make", "PY=python", "all"]
