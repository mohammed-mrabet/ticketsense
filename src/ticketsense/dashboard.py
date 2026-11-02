"""Tableau de bord de suivi du support.

Lancement : streamlit run src/ticketsense/dashboard.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ticketsense import database as db
from ticketsense.config import DB_PATH, REPORTS_DIR
from ticketsense.predictor import get_classifier

st.set_page_config(page_title="TicketSense", page_icon="🎫", layout="wide")


@st.cache_resource
def connection():
    conn = db.connect(DB_PATH)
    if conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] == 0:
        db.load_csv(conn)
    return conn


@st.cache_data
def load_json(name: str):
    path = REPORTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


conn = connection()

st.title("🎫 TicketSense")
st.caption("Classification et analyse des tickets de support informatique")

tab_suivi, tab_incidents, tab_classifier, tab_model = st.tabs(
    ["📊 Suivi", "🚨 Incidents récurrents", "🤖 Classer un ticket", "📈 Performance du modèle"]
)

with tab_suivi:
    s = db.summary(conn)
    sla = db.sla_by_priority(conn)
    sla_global = round((sla["sla_met_pct"] * sla["resolved"]).sum() / sla["resolved"].sum(), 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tickets", f"{s['total']:,}".replace(",", " "))
    c2.metric("Encore ouverts", s["open"])
    c3.metric("Délai moyen de résolution", f"{s['avg_resolution_hours']} h")
    c4.metric("SLA respecté", f"{sla_global} %")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Volume hebdomadaire")
        weekly = db.weekly_volume(conn)
        st.line_chart(weekly, x="week", y="tickets", height=300)
    with right:
        st.subheader("Tickets par catégorie")
        st.bar_chart(db.by_category(conn), x="category", y="tickets", height=300, horizontal=True)

    st.subheader("Délais et SLA par priorité")
    st.dataframe(
        sla.rename(columns={
            "priority": "Priorité", "resolved": "Résolus", "avg_resolution_hours": "Délai moyen (h)",
            "sla_hours": "Objectif SLA (h)", "sla_met_pct": "SLA respecté (%)",
        }),
        hide_index=True,
        use_container_width=True,
    )

with tab_incidents:
    incidents = load_json("incidents.json")
    if not incidents:
        st.info("Lancez `python -m ticketsense.clustering` pour détecter les incidents.")
    else:
        st.write(
            f"**{len(incidents)} pics d'incidents** détectés par clustering (TF-IDF + K-Means) : "
            "des tickets très similaires arrivés en masse le même jour."
        )
        for inc in incidents:
            with st.container(border=True):
                a, b = st.columns([1, 3])
                a.metric(inc["date"], f"{inc['tickets']} tickets", f"habituellement ≤ {inc['usual_daily_volume']:.0f}/jour",
                         delta_color="off")
                b.markdown(f"**{inc['main_category']}** · mots-clés : `{'`, `'.join(inc['keywords'][:5])}`")
                b.caption(f"Exemple : « {inc['example']}… »")

        st.subheader("Volume quotidien par catégorie")
        category = st.selectbox("Catégorie", sorted({i["main_category"] for i in incidents}))
        daily = db.daily_volume(conn, category)
        daily["day"] = pd.to_datetime(daily["day"])
        st.bar_chart(daily, x="day", y="tickets", height=260)

with tab_classifier:
    st.write("Saisissez un ticket : le modèle propose une catégorie et une priorité.")
    with st.form("predict"):
        subject = st.text_input("Objet", "Imprimante")
        body = st.text_area("Description", "L'imprimante du 2e étage est hors ligne, toute l'équipe est bloquée pour imprimer les contrats.")
        submitted = st.form_submit_button("Classer")
    if submitted and body.strip():
        p = get_classifier().predict(subject, body)
        a, b = st.columns(2)
        a.metric("Catégorie", p.category, f"confiance {p.category_confidence:.0%}", delta_color="off")
        b.metric("Priorité", p.priority, f"confiance {p.priority_confidence:.0%}", delta_color="off")
        if p.needs_review:
            st.warning("Confiance faible : ce ticket devrait être vérifié par un technicien.")
        a.bar_chart(pd.Series(p.category_scores, name="probabilité"), horizontal=True, height=220)
        b.bar_chart(pd.Series(p.priority_scores, name="probabilité"), horizontal=True, height=220)

with tab_model:
    metrics = load_json("metrics.json")
    if not metrics:
        st.info("Lancez `python -m ticketsense.train` pour entraîner les modèles.")
    else:
        rows = []
        for target, label in (("category", "Catégorie"), ("priority", "Priorité")):
            for name, key in (("Classe majoritaire (référence)", "baseline_most_frequent"), ("TF-IDF + régression logistique", "tfidf_logreg")):
                m = metrics[target][key]
                rows.append({"Cible": label, "Modèle": name, "Accuracy": m["accuracy"], "Précision": m["precision_macro"],
                             "Rappel": m["recall_macro"], "F1 macro": m["f1_macro"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f"Jeu de test : {metrics['category']['test_size']} tickets (20 %), entraînement le {metrics['trained_at'][:10]}.")
        a, b = st.columns(2)
        a.image(str(REPORTS_DIR / "confusion_category.png"))
        b.image(str(REPORTS_DIR / "confusion_priority.png"))
