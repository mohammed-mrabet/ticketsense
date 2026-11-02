"""Entraînement et évaluation des modèles de catégorie et de priorité.

Usage : python -m ticketsense.train
Produit : models/category.joblib, models/priority.joblib,
          reports/metrics.json, reports/confusion_category.png, reports/confusion_priority.png
"""

from __future__ import annotations

import json
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from . import __version__  # noqa: E402
from .config import CATEGORIES, DATA_PATH, MODELS_DIR, PRIORITIES, REPORTS_DIR  # noqa: E402
from .model import build_pipeline  # noqa: E402
from .preprocessing import ticket_text  # noqa: E402

RANDOM_STATE = 42


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["text"] = [ticket_text(s, b) for s, b in zip(df["subject"], df["body"], strict=True)]
    return df


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
    }


def train_target(df: pd.DataFrame, target: str, labels: list[str]) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df[target], test_size=0.2, stratify=df[target], random_state=RANDOM_STATE
    )

    # Référence naïve : toujours prédire la classe la plus fréquente.
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

    model = build_pipeline().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / f"{target}.joblib")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=labels, ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35
    )
    ax.set_title(f"Matrice de confusion – {target} (jeu de test)")
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"confusion_{target}.png", dpi=120)
    plt.close(fig)

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "baseline_most_frequent": scores(y_test, dummy.predict(X_test)),
        "tfidf_logreg": scores(y_test, y_pred),
        "per_class": classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    metrics = {
        "version": __version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_size": len(df),
        "category": train_target(df, "category", CATEGORIES),
        "priority": train_target(df, "priority", PRIORITIES),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    for target in ("category", "priority"):
        m = metrics[target]
        print(f"\n== {target} ==")
        print(f"  référence (classe majoritaire) : F1 macro = {m['baseline_most_frequent']['f1_macro']:.3f}")
        print(f"  TF-IDF + régression logistique : accuracy = {m['tfidf_logreg']['accuracy']:.3f}"
              f" | F1 macro = {m['tfidf_logreg']['f1_macro']:.3f}")
    print(f"\nModèles enregistrés dans {MODELS_DIR}, rapports dans {REPORTS_DIR}")


if __name__ == "__main__":
    main()
