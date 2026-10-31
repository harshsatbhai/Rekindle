"""Infer the probable cause of disengagement from evidence gathered with the SQL tool.

Each cause gets an evidence score in [0, 1] built from interpretable signals; scores
are normalised into a probability distribution and then updated with what the agent
already learned about this learner (interventions that failed make their target
cause less likely: P(c | intervention i failed) ∝ P(c) * (1 - eff[c][i])).
"""
from __future__ import annotations

import re

CAUSES = ["stuck_on_topic", "low_scores", "financial", "motivation"]
CAUSE_LABEL = {"stuck_on_topic": "Stuck on a topic", "low_scores": "Low scores across modules",
               "financial": "Financial difficulty", "motivation": "Low motivation"}

KW = {
    "stuck": r"can'?t (solve|understand|get)|don'?t understand|still can'?t|explain|confus|stuck|keep failing|lost on",
    "scores": r"scores?|marks|grades?|failing|assessments?|quiz(zes)? (are|is) hard|harder",
    "finance": r"emi|payment|pay|fees?|money|afford|lost my job|financial|loan|salary",
    "motivation": r"bored|boring|busy|losing interest|not sure .* worth|unmotivated|motivation|energy|tired|not enjoying",
    "absence": r"travel|vacation|holiday|wedding|trip|back in|away for|leave from",
}


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _match(key, text):
    return bool(re.search(KW[key], text or "", re.I))


def gather_evidence(tools, learner_id: str, day: int) -> dict:
    """Run the evidence queries. Returns raw evidence plus a list of tool failures (if any)."""
    q = {
        "quiz_by_topic": ("""SELECT topic, COUNT(*) n, ROUND(AVG(score),1) avg, SUM(score < 55) fails, MAX(attempt) max_attempt,
                             MAX(day) last_day FROM quiz_scores WHERE learner_id=? AND day >= ? AND day < ? GROUP BY topic""",
                          (learner_id, day - 21, day)),
        "progress": ("""SELECT topic, started_day, completed_day, time_spent_min, video_rewatches FROM topic_progress
                        WHERE learner_id=? ORDER BY started_day""", (learner_id,)),
        "payments": ("SELECT due_day, paid_day, amount, status FROM payments WHERE learner_id=? AND due_day <= ?", (learner_id, day)),
        "tickets": ("SELECT day, category, text FROM support_tickets WHERE learner_id=? AND day >= ? AND day < ? ORDER BY day",
                    (learner_id, day - 28, day)),
        "surveys": ("SELECT day, motivation, comment FROM surveys WHERE learner_id=? AND day >= ? AND day < ? ORDER BY day",
                    (learner_id, day - 21, day)),
        "logins": ("SELECT day, minutes FROM logins WHERE learner_id=? AND day < ? ORDER BY day", (learner_id, day)),
    }
    ev, failures = {"day": day}, []
    for key, (sql, params) in q.items():
        r = tools.call("sql_query", sql=sql, params=params)
        if r.ok:
            ev[key] = r.data
        else:
            ev[key], _ = [], failures.append({"query": key, "error": r.error})
    ev["failures"] = failures
    return ev


def diagnose(ev: dict, risk: dict, tried: list[str] | None = None, prior_eff: dict | None = None) -> dict:
    day = ev["day"]
    findings: dict[str, list[str]] = {c: [] for c in CAUSES}
    scores = {}

    # ---- stuck on a topic: repeated failures concentrated on one topic, long dwell, rewatches, help requests
    qt = ev.get("quiz_by_topic", [])
    worst = max(qt, key=lambda r: (r["fails"] or 0, -(r["avg"] or 100)), default=None)
    total_fails = sum(r["fails"] or 0 for r in qt)
    open_topics = [p for p in ev.get("progress", []) if p["completed_day"] is None]
    cur = open_topics[-1] if open_topics else None
    dwell = day - cur["started_day"] if cur else 0
    rewatches = cur["video_rewatches"] if cur else 0
    stuck_tickets = [t for t in ev.get("tickets", []) if _match("stuck", t["text"])]
    other = [r for r in qt if worst and r["topic"] != worst["topic"] and r["avg"] is not None]
    others_ok = all(r["avg"] >= 62 for r in other) if other else True
    s = 0.0
    if worst and (worst["fails"] or 0) >= 2:
        conc = (worst["fails"] or 0) / max(total_fails, 1)
        s += .35 * _clip(((worst["fails"] or 0) - 1) / 3) * conc
        findings["stuck_on_topic"].append(f"failed {worst['fails']} of {worst['n']} quiz attempts on {worst['topic']} in the last 3 weeks (average across all attempts {worst['avg']}%)")
    if cur and dwell >= 7:
        s += .2 * _clip((dwell - 6) / 8)
        findings["stuck_on_topic"].append(f"on '{cur['topic']}' for {dwell} days without completing it (typical: ~5 days)")
    if rewatches >= 3:
        s += .15 * _clip(rewatches / 8)
        findings["stuck_on_topic"].append(f"rewatched {cur['topic']} videos {rewatches} times")
    if stuck_tickets:
        s += .25
        findings["stuck_on_topic"].append(f"asked for help: \"{stuck_tickets[-1]['text']}\" (day {stuck_tickets[-1]['day']})")
    if not others_ok:
        s *= .7
    scores["stuck_on_topic"] = s
    stuck_topic = worst["topic"] if worst and (worst["fails"] or 0) >= 2 else (cur["topic"] if cur else None)

    # ---- low scores across modules
    all_avgs = [r["avg"] for r in qt if r["avg"] is not None]
    overall = sum(r["avg"] * r["n"] for r in qt) / max(sum(r["n"] for r in qt), 1) if qt else None
    low_topics = [r["topic"] for r in qt if r["avg"] is not None and r["avg"] < 62]
    s = 0.0
    if overall is not None and overall < 68:
        s += .45 * _clip((70 - overall) / 20)
        findings["low_scores"].append(f"average quiz score {overall:.0f}% over the last 3 weeks")
    if len(low_topics) >= 2:
        s += .3
        findings["low_scores"].append(f"below 62% on {len(low_topics)} different topics ({', '.join(low_topics[:3])})")
    if risk.get("baseline_score") and risk.get("recent_score") and risk["baseline_score"] - risk["recent_score"] > 8:
        s += .1
        findings["low_scores"].append(f"their own quiz average fell from {risk['baseline_score']}% to {risk['recent_score']}%")
    score_tickets = [t for t in ev.get("tickets", []) if t["category"] == "assessment" or (_match("scores", t["text"]) and not _match("stuck", t["text"]))]
    if score_tickets:
        s += .2
        findings["low_scores"].append(f"wrote: \"{score_tickets[-1]['text']}\"")
    scores["low_scores"] = s

    # ---- financial
    bad_pay = [p for p in ev.get("payments", []) if p["status"] in ("failed", "overdue")]
    fin_tickets = [t for t in ev.get("tickets", []) if t["category"] == "billing" or _match("finance", t["text"])]
    s = 0.0
    if bad_pay:
        p = bad_pay[0]
        s += .5
        findings["financial"].append(f"EMI of ₹{p['amount']} due day {p['due_day']} is {p['status']}")
        logins = [l["day"] for l in ev.get("logins", [])]
        before = sum(1 for d in logins if p["due_day"] - 7 <= d < p["due_day"]) / 7
        after_days = max(1, min(7, day - p["due_day"]))
        after = sum(1 for d in logins if p["due_day"] <= d < p["due_day"] + after_days) / after_days
        if before - after >= .4:
            s += .2
            findings["financial"].append(f"activity dropped abruptly right after the due date ({before:.0%} → {after:.0%} of days active)")
    if fin_tickets:
        s += .3
        findings["financial"].append(f"wrote to billing: \"{fin_tickets[-1]['text']}\"")
    scores["financial"] = s

    # ---- motivation
    sv = ev.get("surveys", [])
    s = 0.0
    mot_direct = 0.0  # survey-based part only (trend signals may be explained by a planned absence)
    if sv:
        m = sum(x["motivation"] for x in sv) / len(sv)
        if m < 3.5:
            s += .35 * _clip((3.6 - m) / 2)
            findings["motivation"].append(f"pulse-survey motivation {m:.1f}/5 over the last 3 weeks")
        mc = [x for x in sv if _match("motivation", x["comment"])]
        if mc:
            s += .25
            findings["motivation"].append(f"survey comment: \"{mc[-1]['comment']}\"")
        mot_direct = s
    if risk.get("login_trend_slope", 0) < -.05 and not bad_pay:
        s += .2 * _clip(-risk["login_trend_slope"] / .2)
        findings["motivation"].append(f"weekly login rate steadily declining {risk.get('weekly_login_rate')}")
    if risk.get("minutes_trend_slope", 0) < -3:
        s += .15
        findings["motivation"].append(f"study sessions getting shorter ({risk['minutes_trend_slope']:+.0f} min/week)")
    if overall is not None and overall >= 65 and not bad_pay:
        s += .05  # disengaging despite being able to do the work
    scores["motivation"] = s

    # ---- explained absence (e.g. learner told us they were travelling and is back)
    absence = [t for t in ev.get("tickets", []) if _match("absence", t["text"])]
    recent_active = sum(1 for l in ev.get("logins", []) if l["day"] >= day - 3)
    direct = max(scores["stuck_on_topic"], scores["low_scores"], scores["financial"], mot_direct)
    explained_absence = bool(absence) and recent_active >= 1 and direct < .3

    # ---- normalise + Bayesian update with what memory says has already failed
    floor = .03
    raw = {c: scores[c] + floor for c in CAUSES}
    for itype in tried or []:
        for c in CAUSES:
            raw[c] *= (1 - (prior_eff or {}).get(c, {}).get(itype, 0.0))
    z = sum(raw.values())
    probs = {c: round(raw[c] / z, 3) for c in CAUSES}
    ranked = sorted(probs, key=probs.get, reverse=True)
    top = ranked[0]
    return {
        "cause": top, "probs": probs, "scores": {c: round(v, 3) for c, v in scores.items()},
        "confidence": probs[top], "margin": round(probs[top] - probs[ranked[1]], 3),
        "evidence": findings[top], "all_findings": findings, "stuck_topic": stuck_topic,
        "unclear": max(scores.values()) < .2,
        "explained_absence": explained_absence,
        "absence_note": absence[-1]["text"] if absence else None,
    }
