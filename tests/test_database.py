import pandas as pd
import pytest

from ticketsense import database as db


@pytest.fixture
def conn(tmp_path):
    csv = tmp_path / "tickets.csv"
    pd.DataFrame([
        # Critique résolu en 2 h (SLA 4 h respecté)
        {"ticket_id": "T1", "created_at": "2026-10-05T09:00", "resolved_at": "2026-10-05T11:00", "channel": "Portail",
         "subject": "VPN", "body": "vpn ko", "category": "Réseau", "priority": "Critique"},
        # Critique résolu en 6 h (SLA dépassé)
        {"ticket_id": "T2", "created_at": "2026-10-06T09:00", "resolved_at": "2026-10-06T15:00", "channel": "Portail",
         "subject": "VPN", "body": "vpn ko", "category": "Réseau", "priority": "Critique"},
        # Ticket encore ouvert
        {"ticket_id": "T3", "created_at": "2026-10-12T10:00", "resolved_at": None, "channel": "E-mail",
         "subject": "Toner", "body": "toner vide", "category": "Impression", "priority": "Basse"},
    ]).to_csv(csv, index=False)
    connection = db.connect(tmp_path / "test.db")
    db.load_csv(connection, csv)
    yield connection
    connection.close()


def test_summary(conn):
    assert db.summary(conn) == {"total": 3, "open": 1, "avg_resolution_hours": 4.0}


def test_sla_par_priorite(conn):
    sla = db.sla_by_priority(conn).set_index("priority")
    assert sla.loc["Critique", "resolved"] == 2
    assert sla.loc["Critique", "sla_met_pct"] == 50.0
    assert "Basse" not in sla.index  # le ticket ouvert n'est pas compté


def test_volume_hebdomadaire_par_lundi(conn):
    weekly = db.weekly_volume(conn)
    # 05/10/2026 est un lundi : T1 (lundi) et T2 (mardi) tombent dans la même semaine.
    assert weekly.to_dict("records") == [{"week": "2026-10-05", "tickets": 2}, {"week": "2026-10-12", "tickets": 1}]
