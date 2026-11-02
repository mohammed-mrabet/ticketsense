"""Chargement des modèles entraînés et prédiction sur un nouveau ticket."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from .config import MODELS_DIR
from .preprocessing import ticket_text


@dataclass
class Prediction:
    category: str
    category_confidence: float
    priority: str
    priority_confidence: float
    category_scores: dict[str, float]
    priority_scores: dict[str, float]
    needs_review: bool


class TicketClassifier:
    # En dessous de ce seuil, le ticket est signalé pour une vérification humaine.
    REVIEW_THRESHOLD = 0.55

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.category_model = joblib.load(models_dir / "category.joblib")
        self.priority_model = joblib.load(models_dir / "priority.joblib")

    @staticmethod
    def _scores(model, text: str) -> dict[str, float]:
        probas = model.predict_proba([text])[0]
        ranked = sorted(zip(model.classes_, probas, strict=True), key=lambda x: x[1], reverse=True)
        return {label: round(float(p), 4) for label, p in ranked}

    def predict(self, subject: str | None, body: str) -> Prediction:
        text = ticket_text(subject, body)
        cat_scores = self._scores(self.category_model, text)
        prio_scores = self._scores(self.priority_model, text)
        category, cat_conf = next(iter(cat_scores.items()))
        priority, prio_conf = next(iter(prio_scores.items()))
        return Prediction(
            category=category,
            category_confidence=cat_conf,
            priority=priority,
            priority_confidence=prio_conf,
            category_scores=cat_scores,
            priority_scores=prio_scores,
            needs_review=min(cat_conf, prio_conf) < self.REVIEW_THRESHOLD,
        )


@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    return TicketClassifier()
