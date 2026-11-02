"""Stockage des tickets dans SQLite et indicateurs de suivi calculés en SQL.

Usage : python -m ticketsense.database   (crée data/tickets.db à partir du CSV)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import DATA_PATH, DB_PATH, SLA_HOURS

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    resolved_at TEXT,
    channel     TEXT,
    subject     TEXT,
    body        TEXT NOT NULL,
    category    TEXT NOT NULL,
    priority    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets (created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets (category);
"""

# Délai de résolution en heures (julianday renvoie des jours).
RESOLUTION_HOURS_SQL = "(julianday(resolved_at) - julianday(created_at)) * 24"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    # check_same_thread=False : Streamlit réutilise la connexion depuis plusieurs threads (lecture seule).
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executescript(SCHEMA)
    return conn


def load_csv(conn: sqlite3.Connection, csv_path: Path = DATA_PATH) -> int:
    df = pd.read_csv(csv_path).fillna({"resolved_at": ""})
    df["resolved_at"] = df["resolved_at"].replace("", None)
    conn.execute("DELETE FROM tickets")
    df.to_sql("tickets", conn, if_exists="append", index=False)
    conn.commit()
    return len(df)


def _query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def summary(conn: sqlite3.Connection) -> dict:
    row = conn.execute(f"""
        SELECT COUNT(*),
               SUM(resolved_at IS NULL),
               ROUND(AVG({RESOLUTION_HOURS_SQL}), 1)
        FROM tickets
    """).fetchone()
    return {"total": row[0], "open": row[1] or 0, "avg_resolution_hours": row[2]}


def weekly_volume(conn: sqlite3.Connection) -> pd.DataFrame:
    """Nombre de tickets par semaine (lundi de la semaine)."""
    return _query(conn, """
        SELECT date(created_at, '-6 days', 'weekday 1') AS week, COUNT(*) AS tickets
        FROM tickets
        GROUP BY week
        ORDER BY week
    """)


def by_category(conn: sqlite3.Connection) -> pd.DataFrame:
    return _query(conn, f"""
        SELECT category,
               COUNT(*) AS tickets,
               ROUND(AVG({RESOLUTION_HOURS_SQL}), 1) AS avg_resolution_hours
        FROM tickets
        GROUP BY category
        ORDER BY tickets DESC
    """)


def sla_by_priority(conn: sqlite3.Connection) -> pd.DataFrame:
    """Délai moyen et taux de respect du SLA pour chaque priorité (tickets résolus)."""
    cases = " ".join(f"WHEN '{p}' THEN {h}" for p, h in SLA_HOURS.items())
    return _query(conn, f"""
        SELECT priority,
               COUNT(*) AS resolved,
               ROUND(AVG({RESOLUTION_HOURS_SQL}), 1) AS avg_resolution_hours,
               CASE priority {cases} END AS sla_hours,
               ROUND(100.0 * AVG({RESOLUTION_HOURS_SQL} <= CASE priority {cases} END), 1) AS sla_met_pct
        FROM tickets
        WHERE resolved_at IS NOT NULL
        GROUP BY priority
        ORDER BY sla_hours
    """)


def daily_volume(conn: sqlite3.Connection, category: str | None = None) -> pd.DataFrame:
    sql = "SELECT date(created_at) AS day, COUNT(*) AS tickets FROM tickets"
    params: tuple = ()
    if category:
        sql += " WHERE category = ?"
        params = (category,)
    return _query(conn, sql + " GROUP BY day ORDER BY day", params)


def main() -> None:
    with connect() as conn:
        n = load_csv(conn)
        print(f"{n} tickets chargés dans {DB_PATH}")
        print(summary(conn))
        print(sla_by_priority(conn).to_string(index=False))


if __name__ == "__main__":
    main()
