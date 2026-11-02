"""Détection des incidents récurrents par clustering des tickets.

Idée : des tickets très proches qui arrivent en masse le même jour signalent une panne
générale (ex. VPN en panne pour tout le monde). On regroupe les tickets par similarité
de texte (TF-IDF -> réduction SVD -> K-Means), puis on repère les groupes dont le volume
journalier explose par rapport à leur volume habituel.

Usage : python -m ticketsense.clustering   ->  reports/incidents.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .config import REPORTS_DIR
from .train import load_dataset

# Mots trop fréquents dans les tickets pour décrire un incident.
STOP_WORDS = (
    "bonjour salut hello merci cordialement journee bonne le la les de des du un une et en a au aux je j "
    "mon ma mes ne pas plus sur pour que qui est il elle on nous vous ce cette avec depuis dans tout "
    "toute tous d l n s c qu me m se y ai avoir suis etre fait passage avance aide sur souci probleme "
    "moi chez matin ce depuis quand"
).split()


def cluster_tickets(df: pd.DataFrame, n_clusters: int = 30, random_state: int = 42):
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vectorizer.fit_transform(df["text"])
    lsa = make_pipeline(TruncatedSVD(n_components=100, random_state=random_state), Normalizer(copy=False))
    X_reduced = lsa.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X_reduced)

    # Mots-clés de chaque groupe : centres ramenés dans l'espace TF-IDF.
    centers = lsa.named_steps["truncatedsvd"].inverse_transform(km.cluster_centers_)
    terms = np.array(vectorizer.get_feature_names_out())
    keywords = {i: terms[np.argsort(centers[i])[::-1][:6]].tolist() for i in range(n_clusters)}
    return km.labels_, keywords


def detect_incidents(df: pd.DataFrame, labels: np.ndarray, keywords: dict, min_tickets: int = 15, ratio: float = 3.0) -> list[dict]:
    """Un pic = un jour où un groupe reçoit au moins `min_tickets` tickets et `ratio` fois
    plus que son 90e centile journalier (un lundi chargé ne suffit donc pas à déclencher
    une alerte). Les tickets du lendemain encore anormalement nombreux sont rattachés au même incident."""
    data = df.assign(cluster=labels, day=pd.to_datetime(df["created_at"]).dt.date)
    incidents = []
    for cluster, group in data.groupby("cluster"):
        daily = group.groupby("day").size()
        all_days = pd.date_range(data["day"].min(), data["day"].max()).date
        daily = daily.reindex(all_days, fill_value=0)
        usual = max(1.0, float(daily.quantile(0.9)))
        for i, (day, count) in enumerate(daily.items()):
            if count < max(min_tickets, ratio * usual):
                continue
            total = int(count)
            # Suite de l'incident le lendemain ?
            if i + 1 < len(daily) and daily.iloc[i + 1] > 2 * usual:
                total += int(daily.iloc[i + 1])
            same_day = group[group["day"] == day]
            incidents.append({
                "date": day.isoformat(),
                "cluster": int(cluster),
                "tickets": total,
                "usual_daily_volume": usual,
                "main_category": same_day["category"].mode().iloc[0],
                "keywords": keywords[cluster],
                "example": same_day["body"].iloc[0][:160],
            })
    return sorted(incidents, key=lambda i: i["date"])


def main() -> None:
    df = load_dataset()
    labels, keywords = cluster_tickets(df)
    incidents = detect_incidents(df, labels, keywords)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "incidents.json").write_text(json.dumps(incidents, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(incidents)} pics d'incidents détectés :")
    for inc in incidents:
        print(f"  {inc['date']}  {inc['tickets']:>3} tickets  [{inc['main_category']}]  {', '.join(inc['keywords'][:4])}")


if __name__ == "__main__":
    main()
