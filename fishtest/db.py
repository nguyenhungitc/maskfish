"""SQLite storage of the tests ("runs") and of a few settings.

The web server and the runner process each open their own connection; the
database is in WAL mode so that reads never wait for the runner's writes.
"""

from __future__ import annotations

import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created       REAL NOT NULL,
    updated       REAL NOT NULL,
    started       REAL,
    finished      REAL,
    status        TEXT NOT NULL,          -- pending building running finished stopped failed
    result        TEXT NOT NULL DEFAULT '',   -- passed / failed (SPRT), '' otherwise
    error         TEXT NOT NULL DEFAULT '',
    username      TEXT NOT NULL DEFAULT '',
    info          TEXT NOT NULL DEFAULT '',
    repo          TEXT NOT NULL,
    base_ref      TEXT NOT NULL,
    base_sha      TEXT NOT NULL,
    base_msg      TEXT NOT NULL DEFAULT '',
    new_ref       TEXT NOT NULL,
    new_sha       TEXT NOT NULL,
    new_msg       TEXT NOT NULL DEFAULT '',
    base_net      TEXT NOT NULL DEFAULT '',
    new_net       TEXT NOT NULL DEFAULT '',
    base_options  TEXT NOT NULL DEFAULT '',
    new_options   TEXT NOT NULL DEFAULT '',
    base_make     TEXT NOT NULL DEFAULT '',
    new_make      TEXT NOT NULL DEFAULT '',
    tc            TEXT NOT NULL,
    threads       INTEGER NOT NULL DEFAULT 1,
    hash          INTEGER NOT NULL DEFAULT 16,
    test_type     TEXT NOT NULL,          -- sprt / numgames
    elo0          REAL NOT NULL DEFAULT 0,
    elo1          REAL NOT NULL DEFAULT 2,
    elo_model     TEXT NOT NULL DEFAULT 'normalized',
    alpha         REAL NOT NULL DEFAULT 0.05,
    beta          REAL NOT NULL DEFAULT 0.05,
    num_games     INTEGER NOT NULL DEFAULT 0, -- numgames: games to play; sprt: maximum (0 = none)
    priority      INTEGER NOT NULL DEFAULT 0,
    plies_min     INTEGER NOT NULL DEFAULT 2,
    plies_max     INTEGER NOT NULL DEFAULT 6,
    adjudicate    INTEGER NOT NULL DEFAULT 3000,
    full_info     INTEGER NOT NULL DEFAULT 0,
    seed          INTEGER NOT NULL,
    penta         TEXT NOT NULL DEFAULT '[0, 0, 0, 0, 0]',
    wins          INTEGER NOT NULL DEFAULT 0,
    losses        INTEGER NOT NULL DEFAULT 0,
    draws         INTEGER NOT NULL DEFAULT 0,
    crashes       INTEGER NOT NULL DEFAULT 0,
    time_losses   INTEGER NOT NULL DEFAULT 0,
    llr           REAL NOT NULL DEFAULT 0,
    llr_history   TEXT NOT NULL DEFAULT '[]',
    base_bench    TEXT NOT NULL DEFAULT '',
    new_bench     TEXT NOT NULL DEFAULT '',
    log           TEXT NOT NULL DEFAULT '',
    play_seconds  REAL NOT NULL DEFAULT 0,
    spsa          TEXT NOT NULL DEFAULT '{}'      -- SPSA: settings, parameters and history
);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

JSON_FIELDS = ("penta", "llr_history", "spsa")
ACTIVE = ("building", "running")


class DB:
    def __init__(self, path: str):
        self.path = path
        con = self._connect()
        con.executescript(SCHEMA)
        # Columns added after the first version
        have = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
        if "spsa" not in have:
            con.execute("ALTER TABLE runs ADD COLUMN spsa TEXT NOT NULL DEFAULT '{}'")
        con.close()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=60, isolation_level=None)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        return con

    def execute(self, sql, args=()):
        con = self._connect()
        try:
            return con.execute(sql, args)
        finally:
            con.close()

    def query(self, sql, args=()) -> list[dict]:
        con = self._connect()
        try:
            return [self._decode(dict(r)) for r in con.execute(sql, args).fetchall()]
        finally:
            con.close()

    @staticmethod
    def _decode(row: dict) -> dict:
        for k in JSON_FIELDS:
            if k in row and isinstance(row[k], str):
                row[k] = json.loads(row[k])
        return row

    @staticmethod
    def _encode(fields: dict) -> dict:
        out = {}
        for k, v in fields.items():
            out[k] = json.dumps(v) if k in JSON_FIELDS and not isinstance(v, str) else v
        return out

    # ------------------------------------------------------------ runs

    def insert_run(self, fields: dict) -> int:
        now = time.time()
        fields = self._encode({"created": now, "updated": now, "status": "pending", **fields})
        keys = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        con = self._connect()
        try:
            cur = con.execute(f"INSERT INTO runs ({keys}) VALUES ({marks})", list(fields.values()))
            return cur.lastrowid
        finally:
            con.close()

    def get_run(self, run_id: int) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE id = ?", (run_id,))
        return rows[0] if rows else None

    def update_run(self, run_id: int, only_if_status=None, **fields) -> bool:
        """Updates a run; with only_if_status (a status or a tuple of
        statuses), only when the run still has that status. Returns whether
        the run was updated."""
        fields = self._encode({"updated": time.time(), **fields})
        sets = ", ".join(f"{k} = ?" for k in fields)
        args = list(fields.values()) + [run_id]
        sql = f"UPDATE runs SET {sets} WHERE id = ?"
        if only_if_status:
            st = (only_if_status,) if isinstance(only_if_status, str) else tuple(only_if_status)
            sql += f" AND status IN ({', '.join('?' for _ in st)})"
            args += list(st)
        con = self._connect()
        try:
            return con.execute(sql, args).rowcount > 0
        finally:
            con.close()

    def append_log(self, run_id: int, text: str):
        self.execute("UPDATE runs SET log = log || ?, updated = ? WHERE id = ?",
                     (text, time.time(), run_id))

    def delete_run(self, run_id: int):
        self.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def runs(self, statuses, order="priority DESC, id ASC", limit=None, offset=0,
             light=True) -> list[dict]:
        cols = ("*" if not light else
                ", ".join(c for c in RUN_COLUMNS if c not in ("log", "llr_history", "spsa")))
        sql = (f"SELECT {cols} FROM runs WHERE status IN ({', '.join('?' for _ in statuses)}) "
               f"ORDER BY {order}")
        if limit:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"
        return self.query(sql, tuple(statuses))

    def count(self, statuses) -> int:
        con = self._connect()
        try:
            return con.execute(f"SELECT COUNT(*) FROM runs WHERE status IN "
                               f"({', '.join('?' for _ in statuses)})", tuple(statuses)).fetchone()[0]
        finally:
            con.close()

    def next_pending(self) -> dict | None:
        rows = self.query("SELECT * FROM runs WHERE status = 'pending' "
                          "ORDER BY priority DESC, id ASC LIMIT 1")
        return rows[0] if rows else None

    # ------------------------------------------------------------ config

    def set_config(self, key: str, value):
        self.execute("INSERT INTO config (key, value) VALUES (?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     (key, json.dumps(value)))

    def get_config(self, key: str, default=None):
        rows = self.query("SELECT value FROM config WHERE key = ?", (key,))
        return json.loads(rows[0]["value"]) if rows else default


RUN_COLUMNS = [line.split()[0] for line in SCHEMA.split("CREATE TABLE IF NOT EXISTS runs (")[1]
               .split(");")[0].strip().splitlines() if line.strip() and not line.strip().startswith("--")]
