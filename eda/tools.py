"""The agent's tools. Each is a real capability, not a stub:

1. sql_query        - read-only SQL against the learning platform DB (SQLite), with a
                      table allowlist enforced by SQLite's authorizer
2. run_python       - code execution in an isolated subprocess (AST allowlist, timeout,
                      resource limits, empty env)
3. search_resources - TF-IDF search engine over the learning-resource catalog
4. send_message     - email over SMTP (smtplib), with retry/backoff; optional Slack webhook
5. schedule_session - creates an RFC 5545 calendar invite (.ics) for a mentor/counsellor slot

Every call goes through ToolBox.call(), which times it, catches failures, and
returns a ToolResult the agent can observe and re-plan on.
"""
from __future__ import annotations

import ast
import json
import smtplib
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from email.message import EmailMessage
from pathlib import Path

from . import config

COHORT_START = date(2026, 10, 3)  # simulation day 0


def day_to_date(day: int) -> date:
    return COHORT_START + timedelta(days=day)


class ToolError(Exception):
    pass


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: object = None
    error: str | None = None
    latency_ms: float = 0.0
    attempts: int = 1
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------ 1. SQL
ALLOWED_TABLES = {"learners", "logins", "attendance", "quiz_scores", "topic_progress", "payments",
                  "support_tickets", "surveys"}


def sql_query(sql: str, params: tuple = (), db_path: str | None = None, max_rows: int = 5000) -> list[dict]:
    stmt = sql.strip().rstrip(";")
    if ";" in stmt:
        raise ToolError("refused: only a single statement is allowed")
    if not stmt.lower().startswith(("select", "with")):
        raise ToolError("refused: sql_query is read-only (SELECT/WITH only)")
    uri = f"file:{db_path or config.PLATFORM_DB}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)

    def authorizer(action, arg1, arg2, dbname, source):
        if action == sqlite3.SQLITE_READ and arg1 not in ALLOWED_TABLES:
            return sqlite3.SQLITE_DENY
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION):
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorizer)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(stmt, params).fetchmany(max_rows)
    except sqlite3.DatabaseError as e:
        raise ToolError(f"refused or failed: {e}") from e
    finally:
        conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------------------ 2. code execution
SANDBOX_IMPORTS = {"json", "math", "statistics", "sys", "itertools", "collections"}
SANDBOX_BANNED_CALLS = {"open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals", "vars"}


def _check_code(code: str):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] not in SANDBOX_IMPORTS:
                    raise ToolError(f"refused: import of '{n}' is not allowed in the sandbox")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in SANDBOX_BANNED_CALLS:
            raise ToolError(f"refused: call to '{node.func.id}' is not allowed in the sandbox")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr != "__name__":
            raise ToolError("refused: dunder attribute access is not allowed in the sandbox")


def _limits():  # runs in the child before exec
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    except Exception:
        pass


def run_python(code: str, payload: dict, timeout: float = 8.0) -> object:
    """Execute `code` in a fresh isolated interpreter; `payload` arrives on stdin as JSON,
    the program must print one JSON value to stdout."""
    _check_code(code)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", code], input=json.dumps(payload), capture_output=True,
                                  text=True, timeout=timeout, cwd=tmp, env={}, preexec_fn=_limits)
        except subprocess.TimeoutExpired as e:
            raise ToolError(f"sandbox timeout after {timeout}s") from e
    if proc.returncode != 0:
        raise ToolError(f"sandbox error (exit {proc.returncode}): {proc.stderr.strip()[-400:]}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise ToolError(f"sandbox returned non-JSON output: {proc.stdout[:200]!r}") from e


# ------------------------------------------------------------------------------ 3. search
class ResourceSearch:
    def __init__(self, path=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.docs = json.loads(Path(path or config.RESOURCES_FILE).read_text())
        corpus = [f"{d['title']} {d['title']} {d['topic'] or ''} {d['type']} {d['description']}" for d in self.docs]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.matrix = self.vec.fit_transform(corpus)

    def search(self, query: str, k: int = 3, topic: str | None = None, types: list[str] | None = None) -> list[dict]:
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        hits = []
        for i in scores.argsort()[::-1]:
            d = self.docs[i]
            if scores[i] <= 0:
                break
            if topic and d["topic"] not in (topic, None):
                continue
            if types and d["type"] not in types:
                continue
            hits.append({**d, "score": round(float(scores[i]), 3)})
            if len(hits) == k:
                break
        return hits


# ------------------------------------------------------------------------------ 4. messaging
def send_email(to: str, subject: str, body: str, host=None, port=None, timeout=5.0) -> dict:
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = config.SENDER, to, subject
    msg["Message-ID"] = f"<{uuid.uuid4()}@eda-academy.example>"
    msg.set_content(body)
    try:
        with smtplib.SMTP(host or config.SMTP_HOST, port or config.SMTP_PORT, timeout=timeout) as s:
            s.ehlo()
            reply = s.send_message(msg)
    except (OSError, smtplib.SMTPException) as e:
        raise ToolError(f"SMTP delivery failed: {e}") from e
    if reply:
        raise ToolError(f"SMTP refused recipients: {reply}")
    return {"message_id": msg["Message-ID"], "to": to}


def notify_slack(text: str) -> bool:
    if not config.SLACK_WEBHOOK_URL:
        return False
    req = urllib.request.Request(config.SLACK_WEBHOOK_URL, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status == 200


# ------------------------------------------------------------------------------ 5. calendar
def schedule_session(learner: dict, kind: str, host_role: str, day: int, minutes: int = 30) -> dict:
    start = datetime.combine(day_to_date(day), datetime.min.time()).replace(hour=18, minute=30)
    end = start + timedelta(minutes=minutes)
    uid = f"{uuid.uuid4()}@eda-academy.example"
    fmt = "%Y%m%dT%H%M%S"
    ics = "\r\n".join([
        "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//EDA Academy//Rekindle//EN", "BEGIN:VEVENT",
        f"UID:{uid}", f"DTSTAMP:{datetime.utcnow().strftime(fmt)}Z", f"DTSTART;TZID=Asia/Kolkata:{start.strftime(fmt)}",
        f"DTEND;TZID=Asia/Kolkata:{end.strftime(fmt)}", f"SUMMARY:{kind} with your {host_role}",
        f"ATTENDEE;CN={learner['name']}:mailto:{learner['email']}", "STATUS:TENTATIVE", "END:VEVENT", "END:VCALENDAR", ""])
    config.CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CALENDAR_DIR / f"{learner['learner_id']}-{start.strftime('%Y%m%d')}-{kind.replace(' ', '_')}.ics"
    path.write_text(ics)
    return {"file": str(path), "start": start.isoformat(), "host_role": host_role}


# ------------------------------------------------------------------------------ registry
class ToolBox:
    """Dispatches tool calls with timing, retries for transient failures, and failure capture."""

    RETRYABLE = {"send_message"}

    def __init__(self, logger=None, smtp_port=None, retry_backoff=(0.3, 0.8, 1.5)):
        self.logger = logger or (lambda *a, **k: None)
        self.smtp_port = smtp_port
        self.retry_backoff = retry_backoff
        self._search = None
        self.analytics_src = (Path(__file__).parent / "analytics.py").read_text()
        self.calls: list[ToolResult] = []

    @property
    def search(self):
        if self._search is None:
            self._search = ResourceSearch()
        return self._search

    def _impl(self, name, kw):
        if name == "sql_query":
            return sql_query(kw["sql"], tuple(kw.get("params", ())))
        if name == "run_python":
            return run_python(kw["code"], kw["payload"], kw.get("timeout", 8.0))
        if name == "analytics":  # run_python with the vetted analytics program
            return run_python(self.analytics_src, {"fn": kw["fn"], "args": kw["args"]}, kw.get("timeout", 8.0))
        if name == "search_resources":
            return self.search.search(kw["query"], kw.get("k", 3), kw.get("topic"), kw.get("types"))
        if name == "send_message":
            res = send_email(kw["to"], kw["subject"], kw["body"], port=kw.get("port", self.smtp_port))
            if kw.get("notify"):
                try:
                    res["slack"] = notify_slack(kw["notify"])
                except Exception as e:  # notification is best-effort
                    res["slack_error"] = str(e)
            return res
        if name == "schedule_session":
            return schedule_session(kw["learner"], kw["kind"], kw["host_role"], kw["day"])
        raise ToolError(f"unknown tool {name}")

    def call(self, name: str, **kw) -> ToolResult:
        t0, attempts = time.perf_counter(), 0
        delays = list(self.retry_backoff) if name in self.RETRYABLE else []
        while True:
            attempts += 1
            try:
                data = self._impl(name, kw)
                res = ToolResult(name, True, data, attempts=attempts)
                break
            except ToolError as e:
                transient = not str(e).startswith("refused")
                if transient and delays:
                    time.sleep(delays.pop(0))
                    continue
                res = ToolResult(name, False, error=str(e), attempts=attempts)
                break
            except Exception as e:  # unexpected tool crash: surface as observation, never crash the agent
                res = ToolResult(name, False, error=f"{type(e).__name__}: {e}", attempts=attempts)
                break
        res.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.calls.append(res)
        return res
