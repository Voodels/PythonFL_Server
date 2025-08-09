# fl_client/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Configure pip for better network handling
ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "client.py"]