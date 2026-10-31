"""Choose the intervention with the best expected recovery given the cause distribution.

EU(i) = Σ_c P(c) · eff(c, i) − λ · cost(i)

eff(c, i) is a Beta posterior: an expert prior (PRIOR_EFFECT, worth PRIOR_STRENGTH
pseudo-trials) updated with outcomes the agent has measured in previous cycles
(Memory.effectiveness). Interventions already tried on this learner are excluded,
so a failed attempt naturally leads to a different plan.
"""
from __future__ import annotations

from . import config

PRIOR_EFFECT = {
    "stuck_on_topic": {"doubt_session": .80, "easier_plan": .50, "counsellor_call": .30, "financial_counsellor_call": .05, "motivation_nudge": .15},
    "low_scores":     {"doubt_session": .45, "easier_plan": .75, "counsellor_call": .30, "financial_counsellor_call": .05, "motivation_nudge": .15},
    "financial":      {"doubt_session": .05, "easier_plan": .10, "counsellor_call": .50, "financial_counsellor_call": .80, "motivation_nudge": .05},
    "motivation":     {"doubt_session": .15, "easier_plan": .35, "counsellor_call": .70, "financial_counsellor_call": .05, "motivation_nudge": .50},
}
PRIOR_STRENGTH = 4.0
LAMBDA = 0.02

CATALOG = {
    "doubt_session": {"label": "1:1 doubt-clearing session", "cost": 3, "session": ("Doubt-clearing session", "mentor"),
                      "query": "{topic} doubt clearing worked examples explained step by step"},
    "easier_plan": {"label": "Reduced-pace study plan", "cost": 2, "session": None,
                    "query": "reduced pace study plan revision weak topics {topic} beginner recap"},
    "counsellor_call": {"label": "Counsellor call", "cost": 4, "session": ("Check-in call", "learner counsellor"),
                        "query": "motivation peer study group goal wellbeing counselling alumni working full time"},
    "financial_counsellor_call": {"label": "Financial-aid counsellor call", "cost": 4, "session": ("Fee support call", "financial-aid counsellor"),
                                  "query": "EMI restructuring payment pause scholarship financial hardship"},
    "motivation_nudge": {"label": "Goal-reset nudge + peer group", "cost": 1, "session": None,
                         "query": "peer study group motivation personal goal momentum"},
}


def effect(cause: str, itype: str, learned: dict) -> float:
    s, t = learned.get((cause, itype), (0.0, 0.0))
    p = PRIOR_EFFECT[cause][itype]
    return (p * PRIOR_STRENGTH + s) / (PRIOR_STRENGTH + t)


def choose(probs: dict, tried: list[str], learned: dict, attempts: int) -> dict:
    options = []
    for itype, spec in CATALOG.items():
        if itype in tried:
            continue
        p_success = sum(probs[c] * effect(c, itype, learned) for c in probs)
        options.append({"type": itype, "label": spec["label"], "p_success": round(p_success, 3),
                        "utility": round(p_success - LAMBDA * spec["cost"], 3)})
    options.sort(key=lambda o: -o["utility"])
    if attempts >= config.MAX_ATTEMPTS:
        return {"decision": "escalate", "reason": f"{attempts} interventions tried without recovery", "options": options}
    if not options or options[0]["p_success"] < .2:
        return {"decision": "escalate", "reason": "no remaining intervention has a reasonable chance of working", "options": options}
    return {"decision": "intervene", "choice": options[0], "options": options}
