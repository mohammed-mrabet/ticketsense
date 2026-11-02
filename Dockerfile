FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Entraînement au moment du build : l'image contient des modèles cohérents
# avec la version de scikit-learn installée.
RUN python -m ticketsense.train \
 && python -m ticketsense.clustering \
 && python -m ticketsense.database

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "ticketsense.api:app", "--host", "0.0.0.0", "--port", "8000"]
