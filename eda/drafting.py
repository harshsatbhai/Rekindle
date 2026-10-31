"""Draft learner messages and counsellor-facing explanations.

Uses NVIDIA Nemotron (OpenAI-compatible chat API at integrate.api.nvidia.com) when
NVIDIA_API_KEY is set; otherwise, or on any API error, falls back to deterministic
templates. Both paths go through the same guardrails afterwards, so the LLM is
never trusted blindly.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request

from . import config
from .diagnoser import CAUSE_LABEL
from .planner import CATALOG
from .tools import day_to_date

_llm_disabled_reason = None
_local = threading.local()  # why the most recent LLM call (in this thread) fell back to a template


_rate_lock = threading.Lock()
_next_slot = 0.0


def _wait_for_slot():
    """Space request starts at least 60/RPM seconds apart across all threads."""
    global _next_slot
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _next_slot)
        _next_slot = start + 60.0 / config.LLM_RPM
    time.sleep(max(0.0, start - now))


def last_fallback_reason() -> str | None:
    return getattr(_local, "reason", None)


def llm_status() -> str:
    if config.USE_LLM == "off":
        return "off (EDA_USE_LLM=off)"
    if not config.LLM_API_KEY:
        return "templates (no NVIDIA_API_KEY in environment or .env)"
    if _llm_disabled_reason:
        return f"templates ({_llm_disabled_reason})"
    return f"Nemotron ({config.LLM_MODEL})"


def _complete(system: str, prompt: str, max_tokens: int = 1500) -> str | None:
    global _llm_disabled_reason
    _local.reason = None
    if not llm_status().startswith("Nemotron"):
        _local.reason = "LLM off"
        return None
    body = {"model": config.LLM_MODEL, "max_tokens": max_tokens, "temperature": 0.6,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": config.LLM_THINKING}}
    req = urllib.request.Request(f"{config.LLM_BASE_URL}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                                          "Content-Type": "application/json", "Accept": "application/json"})
    for attempt in range(4):
        _wait_for_slot()
        try:
            with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as r:
                data = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            _local.reason = f"HTTP {e.code}"
            if e.code in (401, 403, 404):  # bad key / model: stop trying for this run
                _llm_disabled_reason = f"LLM HTTP {e.code}, using templates"
                return None
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:  # rate limit / overload: back off and retry
                retry_after = e.headers.get("Retry-After") if e.headers else None
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else 4 * 2 ** attempt)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            _local.reason = f"{type(e).__name__}"  # transient: fall back to the template for this call only
            return None
    else:
        return None
    choice = (data.get("choices") or [{}])[0]
    if choice.get("finish_reason") not in ("stop", None):  # truncated output is not safe to send
        _local.reason = f"finish_reason={choice.get('finish_reason')}"
        return None
    text = (choice.get("message") or {}).get("content") or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()  # strip inline reasoning if present
    if not text:
        _local.reason = "empty response"
    return text or None


# ------------------------------------------------------------------------------ explanations
def explain(learner: dict, risk: dict, diag: dict) -> str:
    top = diag["cause"]
    drivers = [d for d in risk.get("drivers", []) if d["contribution"] > 0.02][:3]
    drv = "; ".join(_driver_text(d) for d in drivers) or "multiple small declines"
    base = (f"{learner['name']} is at risk (score {risk['risk']:.2f}). Main signals: {drv}. "
            f"Most likely cause: {CAUSE_LABEL[top]} ({diag['probs'][top]:.0%} confidence). "
            f"Evidence: {'; '.join(diag['evidence']) or 'weak'}.")
    others = sorted(((c, p) for c, p in diag["probs"].items() if c != top), key=lambda x: -x[1])
    if others and others[0][1] > .2:
        base += f" Alternative: {CAUSE_LABEL[others[0][0]]} ({others[0][1]:.0%})."
    llm = _complete(
        "You write concise, factual case notes for learner-success counsellors. Use only the facts given and keep "
        "every number exactly as stated. All comparisons are against the learner's OWN earlier behaviour, never a "
        "cohort. Three sentences maximum, no bullet points, no speculation beyond the evidence.",
        f"Rewrite this case note for a counsellor:\n{base}")
    return llm or base


def _driver_text(d):
    s = d["signal"]
    if s == "login_rate":
        return f"active on {d['recent']:.0%} of days this week vs their usual {d['baseline']:.0%}"
    if s == "attendance_rate":
        return f"live-class attendance {d['recent']:.0%} vs their usual {d['baseline']:.0%}"
    if s == "quiz_score":
        return f"own quiz average {d['recent']}% recently vs {d['baseline']}% earlier"
    if s == "days_since_login":
        return f"{d['recent']} days since last login"
    if s == "low_absolute_activity":
        return f"active on only {d['recent']:.0%} of days"
    if s == "repeated_quiz_failures":
        return f"{d['recent']} failed attempts on one topic in 10 days"
    return "weekly logins trending down"


# ------------------------------------------------------------------------------ learner messages
def _resources_block(resources):
    return "\n".join(f"  • {r['title']} ({r['minutes']} min): {r['url']}" for r in resources)


def template(itype: str, learner: dict, diag: dict, resources: list, session: dict | None) -> tuple[str, str]:
    first = learner["name"].split()[0]
    topic = diag.get("stuck_topic") or "the current module"
    when = ""
    if session:
        when = f"{day_to_date(session['day']).strftime('%A, %d %b')} at 6:30 pm IST"
    res = _resources_block(resources)
    if itype == "doubt_session":
        return (f"Let's crack {topic} together",
                f"Hi {first},\n\nI noticed {topic} has been a tough one. You've put real effort into it, and that's exactly "
                f"when a quick conversation helps most. I've reserved a 30-minute 1:1 doubt-clearing session with a mentor "
                f"on {when}. Bring any problem you're stuck on.\n\nIf you'd like to warm up beforehand:\n{res}\n\n"
                f"Reply to this email if the time doesn't work and we'll find another slot.\n\nCheers,\nLearner Success Team")
    if itype == "easier_plan":
        return ("A lighter plan to get you back on track",
                f"Hi {first},\n\nThe last few modules have moved fast. To make things more manageable, here's a reduced-pace "
                f"plan: about 6 hours a week, with short revision checkpoints before moving on. Nothing you've done "
                f"is lost, and you can switch back to the regular pace any time.\n\nSuggested starting points:\n{res}\n\n"
                f"Reply 'yes' and we'll switch your schedule, or tell us what would work better.\n\nCheers,\nLearner Success Team")
    if itype == "counsellor_call":
        return ("Quick check-in?",
                f"Hi {first},\n\nWe haven't seen you around as much lately, and we wanted to check in. No pressure, just a "
                f"friendly 20-minute call with your learner counsellor to talk about how the course fits with everything "
                f"else going on. I've pencilled in {when}.\n\nSome learners have found these helpful:\n{res}\n\n"
                f"Reply to reschedule or if you'd rather chat over email.\n\nWarmly,\nLearner Success Team")
    if itype == "financial_counsellor_call":
        return ("Options to keep learning while things are tight",
                f"Hi {first},\n\nThanks for letting us know things are difficult right now. Your access stays open, and you have "
                f"options: EMIs can be restructured or paused for up to 60 days, and need-based scholarships are "
                f"available. A financial-aid counsellor can walk you through them, in confidence, on {when}.\n\n"
                f"Details:\n{res}\n\nReply anytime if another time suits you better.\n\nWarmly,\nLearner Success Team")
    return ("A small goal for the next two weeks",
            f"Hi {first},\n\nHow about a small reset? Pick one module to finish in the next two weeks, and join a peer "
            f"study group that meets twice a week online. Having company makes it much easier to keep going.\n\n{res}\n\n"
            f"Reply and tell us your goal, and we'll cheer you on.\n\nCheers,\nLearner Success Team")


def draft(itype: str, learner: dict, diag: dict, resources: list, session: dict | None) -> dict:
    subject, body = template(itype, learner, diag, resources, session)
    llm = _complete(
        "You are a warm, concise learner-success coach at an online academy. Rewrite the draft email to feel personal "
        "and human. Rules: keep every fact, link, date and time exactly; under 170 words; never mention risk scores, "
        "monitoring, algorithms or being 'flagged'; never refer to tracked behaviour (logins, video views, quiz "
        "scores, attendance); no threats, guilt, deadlines or promises of jobs; do not mention money unless the "
        "draft already does. Output only the email body, starting with the greeting.",
        f"Learner first name: {learner['name'].split()[0]}\nIntervention: {CATALOG[itype]['label']}\n\nDraft:\n{body}")
    return {"subject": subject, "body": llm or body, "drafted_by": "nemotron" if llm else f"template ({last_fallback_reason() or 'LLM unavailable'})"}
