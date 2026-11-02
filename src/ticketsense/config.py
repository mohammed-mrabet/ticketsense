"""Chemins du projet, surchargeables par variables d'environnement (utile en Docker)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = Path(os.getenv("TICKETSENSE_DATA", ROOT / "data" / "tickets.csv"))
MODELS_DIR = Path(os.getenv("TICKETSENSE_MODELS", ROOT / "models"))
REPORTS_DIR = Path(os.getenv("TICKETSENSE_REPORTS", ROOT / "reports"))
DB_PATH = Path(os.getenv("TICKETSENSE_DB", ROOT / "data" / "tickets.db"))

CATEGORIES = ["Accès & comptes", "Impression", "Logiciel", "Matériel", "Messagerie", "Réseau"]
PRIORITIES = ["Basse", "Moyenne", "Haute", "Critique"]

# Délai cible de résolution (SLA) en heures, par priorité.
SLA_HOURS = {"Critique": 4, "Haute": 8, "Moyenne": 24, "Basse": 72}
