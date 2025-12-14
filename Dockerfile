FROM python:3.11-slim

# ffmpeg потрібен для mp3
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# 1 воркер (модель важка); таймаут більший на перший прогрів
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-t", "180", "-b", "0.0.0.0:8080", "app:app"]
