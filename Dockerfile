FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY SRC ./SRC
COPY API ./API
COPY MODELS ./MODELS

EXPOSE 8000

CMD ["uvicorn", "API.main:app", "--host", "0.0.0.0", "--port", "8000"]