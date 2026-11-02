"""Tests de l'API. Les modèles doivent être entraînés : python -m ticketsense.train"""

import pytest
from fastapi.testclient import TestClient

from ticketsense.api import app
from ticketsense.config import MODELS_DIR

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "category.joblib").exists(), reason="modèles non entraînés"
)
client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("L'imprimante du 2e étage est hors ligne", "Impression"),
        ("Mon mot de passe a expiré, je ne peux plus me connecter", "Accès & comptes"),
        ("Outlook ne reçoit plus aucun mail depuis ce matin", "Messagerie"),
        ("Le VPN refuse de se connecter depuis chez moi", "Réseau"),
    ],
)
def test_predict_categorie(body, expected):
    response = client.post("/predict", json={"body": body})
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == expected
    assert 0 <= data["category_confidence"] <= 1
    assert abs(sum(data["priority_scores"].values()) - 1) < 1e-3


def test_predict_priorite_critique():
    data = client.post("/predict", json={"body": "Le serveur ERP est en panne, la production est à l'arrêt, toute l'équipe est bloquée"}).json()
    assert data["priority"] == "Critique"


def test_validation_corps_vide():
    assert client.post("/predict", json={"body": ""}).status_code == 422


def test_batch():
    response = client.post("/predict/batch", json={"tickets": [{"body": "toner vide"}, {"body": "écran noir"}]})
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_metrics():
    data = client.get("/metrics").json()
    assert data["category"]["model"]["f1_macro"] > data["category"]["baseline"]["f1_macro"]
