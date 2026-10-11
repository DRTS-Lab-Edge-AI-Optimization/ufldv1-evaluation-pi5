FROM --platform=linux/arm64 python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 procps && rm -rf /var/lib/apt/lists/*
COPY requirements-rpi.txt /app/requirements-rpi.txt
RUN python -m pip install --upgrade pip && python -m pip install -r requirements-rpi.txt
EXPOSE 8888
CMD ["sh", "-c", "exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.token=\"$JUPYTER_TOKEN\""]
