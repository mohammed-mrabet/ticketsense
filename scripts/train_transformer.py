"""Fine-tuning d'un modèle Transformer (DistilBERT multilingue) pour la catégorie des tickets.

Objectif : comparer un modèle de langue pré-entraîné à la référence TF-IDF + régression logistique,
sur exactement le même découpage entraînement / test (même graine aléatoire).

Exécution conseillée sur Google Colab (GPU gratuit, ~5 minutes) :
    !git clone https://github.com/mohammed-mrabet/ticketsense.git && cd ticketsense
    !pip install -r requirements-transformer.txt
    !python scripts/train_transformer.py --target category --epochs 3

Résultat : reports/transformer_<cible>.json (mêmes métriques que la référence).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ticketsense.config import REPORTS_DIR  # noqa: E402
from ticketsense.train import RANDOM_STATE, load_dataset, scores  # noqa: E402

MODEL_NAME = "distilbert-base-multilingual-cased"


class TicketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["category", "priority"], default="category")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    df = load_dataset()
    # Texte brut (avec accents et ponctuation) : le tokenizer du Transformer les exploite.
    texts = (df["subject"].fillna("") + " . " + df["body"]).tolist()
    labels = sorted(df[args.target].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    y = df[args.target].map(label2id).tolist()

    # Même découpage que la référence (stratifié, random_state=42).
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, stratify=df[args.target], random_state=RANDOM_STATE
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encode = lambda xs: tokenizer(xs, truncation=True, max_length=args.max_length)  # noqa: E731
    train_ds = TicketDataset(encode(X_train), y_train)
    test_ds = TicketDataset(encode(X_test), y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label={i: label for label, i in label2id.items()},
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(ROOT / "models" / f"transformer_{args.target}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_strategy="no",
        logging_steps=50,
        report_to="none",
        seed=RANDOM_STATE,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    y_pred = np.argmax(logits, axis=1)
    result = {
        "model": MODEL_NAME,
        "target": args.target,
        "epochs": args.epochs,
        "test_size": len(y_test),
        "scores": scores([labels[i] for i in y_test], [labels[i] for i in y_pred]),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"transformer_{args.target}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["scores"], indent=2))
    print(f"Résultats enregistrés dans {out}")


if __name__ == "__main__":
    main()
