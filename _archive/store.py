"""Alert workflow storage (SQLite, built into Python): New / Acknowledged / Resolved / False positive."""
import os
import sqlite3
from datetime import datetime

DB = "data/alerts.db"
STATUSES = ["New", "Acknowledged", "Resolved", "False positive"]


def _conn():
    os.makedirs("data", exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS alert_status(alert_id TEXT PRIMARY KEY, status TEXT, note TEXT, updated_at TEXT)")
    return c


def get_all():
    with _conn() as c:
        return {r[0]: (r[1], r[2]) for r in c.execute("SELECT alert_id, status, note FROM alert_status")}


def set_status(ids, status, note=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with _conn() as c:
        c.executemany("INSERT OR REPLACE INTO alert_status VALUES (?,?,?,?)", [(i, status, note, now) for i in ids])


def reset():
    with _conn() as c:
        c.execute("DELETE FROM alert_status")
