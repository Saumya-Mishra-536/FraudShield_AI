FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
