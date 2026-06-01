FROM python:3.10-slim

# Установка системных утилит, необходимых для сборки некоторых библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Защита от медленного интернета при сборке образа
ENV PIP_DEFAULT_TIMEOUT=1000

# Оптимизация: ставим легкий PyTorch (CPU) отдельно
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Кэшируем и устанавливаем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порты для Streamlit (8501) и Airflow (8080)
EXPOSE 8501 8080