# Базовый образ: Python 3.11 в облегчённой сборке
FROM python:3.11-slim

WORKDIR /app

# Сначала только зависимости — Docker закеширует этот слой
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Теперь код, интерфейс и сама модель
COPY main.py .
COPY llm_analyzer.py .
COPY index.html .
COPY model.pkl .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
