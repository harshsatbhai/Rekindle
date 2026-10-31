"""Trend analytics executed inside the code-execution sandbox (tools.run_python).

Deliberately stdlib-only and self-contained: the sandbox runs this file's source
in an isolated interpreter (`python -I`) and talks to it over stdin/stdout JSON.
It can also be imported directly (the evaluation harness does this for speed).
"""
import json
import sys

MIN_HISTORY_DAYS = 10


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _window_rate(active_days, a, b, start):
    a = max(a, start)
    if b <= a:
        return None
    return sum(1 for d in active_days if a <= d < b) / (b - a)


def _att_rate(att, a, b):
    xs = [v for d, v in att if a <= d < b]
    return _mean(xs)


def _scores(quizzes, a, b):
    return [q[2] for q in quizzes if a <= q[0] < b]


def _slope(ys):
    n = len(ys)
    if n < 2:
        return 0.0
    xbar, ybar = (n - 1) / 2, sum(ys) / n
    num = sum((i - xbar) * (y - ybar) for i, y in enumerate(ys))
    den = sum((i - xbar) ** 2 for i in range(n))
    return num / den


def risk(day, enrolled_day, logins, attendance, quizzes):
    """Score disengagement risk on `day` from the trailing 28 days.

    logins: [[day, minutes], ...]; attendance: [[day, attended]]; quizzes: [[day, topic, score, attempt]]
    """
    history = day - enrolled_day
    if history < MIN_HISTORY_DAYS:
        return {"status": "insufficient_data", "history_days": history, "risk": None}

    active = sorted({d for d, _ in logins if d < day})
    base_a, base_b = max(enrolled_day, day - 28), day - 14
    if base_b - base_a < 5:
        base_b = day - 7
    rec_a = day - 7

    # personal baseline: the learner's best two of the three weeks before the most recent one
    prior_weeks = [_window_rate(active, day - 7 * (k + 1), day - 7 * k, enrolled_day) for k in (1, 2, 3)]
    prior_weeks = sorted([w for w in prior_weeks if w is not None], reverse=True)[:2]
    base_login = _mean(prior_weeks) if prior_weeks else (_window_rate(active, base_a, base_b, enrolled_day) or 0.0)
    onboarding = _window_rate(active, enrolled_day, min(enrolled_day + 14, day - 7), enrolled_day)
    if onboarding is not None and enrolled_day + 14 <= day - 7:
        base_login = max(base_login, onboarding)  # guards against a baseline that drifted down with the learner
    rec_login = _window_rate(active, rec_a, day, enrolled_day) or 0.0
    last3 = _window_rate(active, day - 3, day, enrolled_day) or 0.0
    base_att = _att_rate(attendance, base_a, base_b)
    rec_att = _att_rate(attendance, day - 10, day)
    s_base = _mean(_scores(quizzes, base_a, day - 10))
    s_rec = _mean(_scores(quizzes, day - 10, day))
    if s_rec is None:
        prior = [q[2] for q in quizzes if q[0] < day]
        s_rec = _mean(prior[-3:]) if prior else None
    last_login = max(active) if active else enrolled_day - 1
    days_since = day - 1 - last_login

    weekly = [_window_rate(active, day - 7 * (k + 1), day - 7 * k, enrolled_day) for k in range(3, -1, -1)]
    weekly = [w for w in weekly if w is not None]
    mins_by_week = []
    for k in range(3, -1, -1):
        ms = [m for d, m in logins if day - 7 * (k + 1) <= d < day - 7 * k]
        if ms:
            mins_by_week.append(sum(ms) / len(ms))

    d_login = _clip((base_login - rec_login) / max(base_login, .2))
    d_att = _clip((base_att - rec_att) / max(base_att, .2)) if (base_att is not None and rec_att is not None) else d_login
    if s_rec is None:
        score_comp = 0.0
    else:
        drop = (s_base - s_rec) / 25 if s_base is not None else 0.0
        score_comp = max(_clip(drop), _clip((62 - s_rec) / 25))
    inactivity = _clip((days_since - 2) / 7)
    trend = _clip(-_slope(weekly) / .15)
    abs_low = _clip((.5 - rec_login) / .35)  # low engagement in absolute terms, whatever the personal history
    fails = {}
    for q in quizzes:
        if day - 10 <= q[0] < day and q[2] < 55:
            fails[q[1]] = fails.get(q[1], 0) + 1
    max_fails = max(fails.values()) if fails else 0
    struggle = _clip((max_fails - 1) / 3)

    raw = .28 * d_login + .2 * d_att + .25 * score_comp + .15 * inactivity + .12 * trend + .15 * struggle + .1 * abs_low
    rebound = last3 >= .67 and d_login > 0 and score_comp < .3
    r = raw * (.55 if rebound else 1.0)

    drivers = [
        {"signal": "login_rate", "baseline": round(base_login, 2), "recent": round(rec_login, 2), "contribution": round(.28 * d_login, 3)},
        {"signal": "attendance_rate", "baseline": None if base_att is None else round(base_att, 2),
         "recent": None if rec_att is None else round(rec_att, 2), "contribution": round(.2 * d_att, 3)},
        {"signal": "quiz_score", "baseline": None if s_base is None else round(s_base, 1),
         "recent": None if s_rec is None else round(s_rec, 1), "contribution": round(.25 * score_comp, 3)},
        {"signal": "days_since_login", "baseline": None, "recent": days_since, "contribution": round(.15 * inactivity, 3)},
        {"signal": "weekly_login_trend", "baseline": None, "recent": None, "contribution": round(.12 * trend, 3)},
        {"signal": "low_absolute_activity", "baseline": .5, "recent": round(rec_login, 2), "contribution": round(.1 * abs_low, 3)},
        {"signal": "repeated_quiz_failures", "baseline": None, "recent": max_fails, "contribution": round(.15 * struggle, 3)},
    ]
    drivers.sort(key=lambda x: -x["contribution"])
    return {
        "status": "ok", "history_days": history, "risk": round(r, 3), "raw_risk": round(raw, 3),
        "rebound_dampened": rebound, "drivers": drivers,
        "weekly_login_rate": [round(w, 2) for w in weekly], "login_trend_slope": round(_slope(weekly), 3),
        "minutes_trend_slope": round(_slope(mins_by_week), 2), "days_since_login": days_since,
        "recent_score": None if s_rec is None else round(s_rec, 1), "baseline_score": None if s_base is None else round(s_base, 1),
    }


def engagement(a, b, enrolled_day, logins, attendance, quizzes):
    """Engagement index in [0, 1] over window [a, b): 0.4 logins + 0.3 attendance + 0.3 scores."""
    active = {d for d, _ in logins}
    lr = _window_rate(active, a, b, enrolled_day)
    if lr is None:
        return {"index": None}
    ar = _att_rate(attendance, a, b)
    sc = _mean(_scores(quizzes, a, b))
    parts = [(.4, lr), (.3, ar), (.3, None if sc is None else sc / 100)]
    w = sum(p[0] for p in parts if p[1] is not None)
    idx = sum(p[0] * p[1] for p in parts if p[1] is not None) / w
    return {"index": round(idx, 3), "login_rate": round(lr, 2), "attendance_rate": None if ar is None else round(ar, 2),
            "avg_score": None if sc is None else round(sc, 1), "window": [a, b]}


FUNCS = {"risk": risk, "engagement": engagement}

if __name__ == "__main__":
    req = json.loads(sys.stdin.read())
    out = FUNCS[req["fn"]](**req["args"])
    sys.stdout.write(json.dumps(out))
