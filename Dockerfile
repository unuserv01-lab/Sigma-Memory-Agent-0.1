FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard]

# Copy source
COPY sma/ ./sma/
COPY api/ ./api/

# Persistent volume for SQLite (mount /data on FC)
RUN mkdir -p /data
ENV SMA_DB_PATH=/data/sma_memory.db

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
