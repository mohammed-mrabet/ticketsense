"""API REST de prédiction.

Lancement : uvicorn ticketsense.api:app --reload
Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import json
from dataclasses import asdict

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .config import REPORTS_DIR
from .predictor import TicketClassifier, get_classifier

app = FastAPI(
    title="TicketSense API",
    version=__version__,
    description="Classification automatique des tickets de support (catégorie et priorité).",
)


class TicketIn(BaseModel):
    subject: str | None = Field(default=None, max_length=200, examples=["VPN impossible"])
    body: str = Field(min_length=3, max_length=5000, examples=["Le VPN refuse de se connecter depuis chez moi, je suis bloqué."])


class PredictionOut(BaseModel):
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    category_scores: dict[str, float]
    priority_scores: dict[str, float]
    needs_review: bool = Field(description="Vrai si le modèle hésite : à vérifier par un technicien")


class BatchIn(BaseModel):
    tickets: list[TicketIn] = Field(min_length=1, max_length=500)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/predict", response_model=PredictionOut)
def predict(ticket: TicketIn, classifier: TicketClassifier = Depends(get_classifier)) -> PredictionOut:
    return PredictionOut(**asdict(classifier.predict(ticket.subject, ticket.body)))


@app.post("/predict/batch", response_model=list[PredictionOut])
def predict_batch(batch: BatchIn, classifier: TicketClassifier = Depends(get_classifier)) -> list[PredictionOut]:
    return [PredictionOut(**asdict(classifier.predict(t.subject, t.body))) for t in batch.tickets]


@app.get("/metrics")
def metrics() -> dict:
    """Scores des modèles mesurés sur le jeu de test lors du dernier entraînement."""
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Modèles non entraînés : lancez python -m ticketsense.train")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        target: {"baseline": data[target]["baseline_most_frequent"], "model": data[target]["tfidf_logreg"]}
        for target in ("category", "priority")
    }
