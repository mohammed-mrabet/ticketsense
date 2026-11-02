"""Définition du modèle de référence : TF-IDF (mots + caractères) puis régression logistique."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def build_pipeline(C: float = 4.0) -> Pipeline:
    features = FeatureUnion([
        # n-grammes de mots : « mot de passe », « hors ligne »...
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        # n-grammes de caractères : robustes aux fautes de frappe (« imprimnte », « outlok »).
        ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=60_000)),
    ])
    classifier = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", classifier)])
