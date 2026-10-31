"""Persistent agent memory (SQLite). Survives process restarts, so state carries
across agent cycles (sessions): learner status, intervention history, measured
outcomes, learned intervention effectiveness, and a full step trace."""
from __future__ import annotations

import json
import sqlite3
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS trace(id INTEGER PRIMARY KEY AUTOINCREMENT, cycle INT, day INT, learner_id TEXT,
    phase TEXT, title TEXT, detail TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS learner_state(learner_id TEXT PRIMARY KEY, status TEXT, risk REAL, cause TEXT,
    cause_probs TEXT, explanation TEXT, evidence TEXT, last_checked_day INT, attempts INT DEFAULT 0,
    last_contact_day INT, ruled_out TEXT DEFAULT '[]', updated REAL);
CREATE TABLE IF NOT EXISTS interventions(id INTEGER PRIMARY KEY AUTOINCREMENT, learner_id TEXT, cause TEXT,
    type TEXT, rationale TEXT, subject TEXT, body TEXT, resources TEXT, status TEXT, created_day INT,
    reviewer TEXT, reviewer_note TEXT, decided_day INT, sent_day INT, followup_day INT, session_file TEXT,
    baseline TEXT, outcome TEXT, success INT, guardrail TEXT, drafted_by TEXT, session_day INT, explanation TEXT);
CREATE TABLE IF NOT EXISTS effectiveness(cause TEXT, type TEXT, successes REAL, trials REAL, PRIMARY KEY(cause, type));
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


class Memory:
    def __init__(self, path=None):
        self.path = str(path or config.MEMORY_DB)
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with self._c() as c:
            c.executescript(SCHEMA)

    def _c(self):
        c = sqlite3.connect(self.path, timeout=30)  # tolerate concurrent writers (parallel episodes)
        c.row_factory = sqlite3.Row
        return c

    def reset(self):
        with self._c() as c:
            for t in ("trace", "learner_state", "interventions", "effectiveness", "meta"):
                c.execute(f"DROP TABLE IF EXISTS {t}")
            c.executescript(SCHEMA)

    # ---------------------------------------------------------------- meta / cycles
    def get_meta(self, key, default=None):
        with self._c() as c:
            r = c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def set_meta(self, key, value):
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, json.dumps(value)))

    def next_cycle(self) -> int:
        n = self.get_meta("cycle", 0) + 1
        self.set_meta("cycle", n)
        return n

    # ---------------------------------------------------------------- trace
    def log(self, cycle, day, learner_id, phase, title, detail=None):
        with self._c() as c:
            c.execute("INSERT INTO trace(cycle, day, learner_id, phase, title, detail, ts) VALUES(?,?,?,?,?,?,?)",
                      (cycle, day, learner_id, phase, title, json.dumps(detail, default=str) if detail is not None else None, time.time()))

    def trace(self, learner_id=None, limit=500):
        q, args = "SELECT * FROM trace", []
        if learner_id:
            q, args = q + " WHERE learner_id=?", [learner_id]
        with self._c() as c:
            rows = c.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows][::-1]

    # ---------------------------------------------------------------- learner state
    def state(self, learner_id) -> dict:
        with self._c() as c:
            r = c.execute("SELECT * FROM learner_state WHERE learner_id=?", (learner_id,)).fetchone()
        if not r:
            return {"learner_id": learner_id, "status": "monitoring", "attempts": 0, "ruled_out": [], "last_contact_day": None}
        d = dict(r)
        for k in ("cause_probs", "evidence", "ruled_out"):
            d[k] = json.loads(d[k]) if d.get(k) else ([] if k == "ruled_out" else None)
        return d

    def save_state(self, learner_id, **fields):
        cur = self.state(learner_id)
        cur.update(fields)
        row = {k: cur.get(k) for k in ("status", "risk", "cause", "cause_probs", "explanation", "evidence",
                                       "last_checked_day", "attempts", "last_contact_day", "ruled_out")}
        for k in ("cause_probs", "evidence", "ruled_out"):
            row[k] = json.dumps(row[k], default=str)
        with self._c() as c:
            c.execute("""INSERT OR REPLACE INTO learner_state(learner_id, status, risk, cause, cause_probs, explanation,
                evidence, last_checked_day, attempts, last_contact_day, ruled_out, updated) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (learner_id, *row.values(), time.time()))

    def all_states(self) -> list[dict]:
        with self._c() as c:
            ids = [r[0] for r in c.execute("SELECT learner_id FROM learner_state ORDER BY learner_id")]
        return [self.state(i) for i in ids]

    # ---------------------------------------------------------------- interventions
    def add_intervention(self, **f) -> int:
        f = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in f.items()}
        cols = ",".join(f)
        with self._c() as c:
            cur = c.execute(f"INSERT INTO interventions({cols}) VALUES({','.join('?' * len(f))})", tuple(f.values()))
            return cur.lastrowid

    def update_intervention(self, iid, **f):
        f = {k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v) for k, v in f.items()}
        with self._c() as c:
            c.execute(f"UPDATE interventions SET {','.join(k + '=?' for k in f)} WHERE id=?", (*f.values(), iid))

    def interventions(self, learner_id=None, status=None) -> list[dict]:
        q, args = "SELECT * FROM interventions WHERE 1=1", []
        if learner_id:
            q += " AND learner_id=?"; args.append(learner_id)
        if status:
            statuses = [status] if isinstance(status, str) else list(status)
            q += f" AND status IN ({','.join('?' * len(statuses))})"; args += statuses
        with self._c() as c:
            rows = c.execute(q + " ORDER BY id", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("resources", "baseline", "outcome", "guardrail"):
                d[k] = json.loads(d[k]) if d.get(k) else None
            out.append(d)
        return out

    def intervention(self, iid) -> dict | None:
        rows = [i for i in self.interventions() if i["id"] == iid]
        return rows[0] if rows else None

    def delivered(self) -> dict[str, list[tuple[str, int]]]:
        """Interventions that actually reached learners (consumed by the world simulator)."""
        out: dict[str, list] = {}
        for i in self.interventions(status=["sent", "measured"]):
            out.setdefault(i["learner_id"], []).append((i["type"], i["sent_day"]))
        return out

    # ---------------------------------------------------------------- learned effectiveness
    def effectiveness(self) -> dict[tuple[str, str], tuple[float, float]]:
        with self._c() as c:
            return {(r["cause"], r["type"]): (r["successes"], r["trials"]) for r in c.execute("SELECT * FROM effectiveness")}

    def record_outcome(self, cause, itype, success: bool):
        s, t = self.effectiveness().get((cause, itype), (0.0, 0.0))
        with self._c() as c:
            c.execute("INSERT OR REPLACE INTO effectiveness VALUES(?,?,?,?)", (cause, itype, s + float(success), t + 1))
