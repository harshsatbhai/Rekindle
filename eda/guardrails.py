"""Guardrails: contact policy, message-content policy, and the approval gate.

Nothing reaches a learner unless (a) contact policy allows it, (b) the final text
passes the content policy, and (c) a named human reviewer approved it. The send
path checks all three again at execution time, so a UI bug cannot bypass them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config


class GuardrailViolation(Exception):
    def __init__(self, report):
        super().__init__("; ".join(v["detail"] for v in report.violations))
        self.report = report


@dataclass
class Report:
    allowed: bool = True
    violations: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def block(self, rule, detail):
        self.allowed = False
        self.violations.append({"rule": rule, "detail": detail})

    def warn(self, rule, detail):
        self.warnings.append({"rule": rule, "detail": detail})

    def as_dict(self):
        return {"allowed": self.allowed, "violations": self.violations, "warnings": self.warnings}


CONTENT_RULES = [
    ("coercion", r"\b(expel\w*|terminat\w* your|cancel\w* your (enrol\w*|access|seat|admission)|legal action|"
                 r"last warning|final warning|or else|blacklist\w*|we will (remove|suspend) you|suspend\w* your)\b",
     "threatening or coercive language"),
    ("shaming", r"\b(lazy|disappointing|not cut out|giving up on yourself|pathetic|you failed|you are failing|careless)\b",
     "shaming or judgemental language"),
    ("false_promise", r"\bguarantee\w*\b.{0,30}\b(job|placement|pass|salary|offer)\b|\b100% (job|placement)\b",
     "promises outcomes the platform cannot guarantee"),
    ("clinical", r"\b(depress\w*|anxiety disorder|mental illness|diagnos\w*)\b",
     "clinical/medical language: leave this to qualified counsellors"),
    ("surveillance", r"\b(risk score|at[- ]risk|flagged|our algorithm|model predicted|tracking you|monitoring you)\b",
     "reveals internal risk labels to the learner (stigmatising)"),
    ("tracked_behaviour", r"\b(rewatch\w*|re-watch\w*|video views|login (data|history|activity)|we track\w*|our records show)\b",
     "refers to tracked behaviour the learner didn't share with us"),
]


def check_content(text: str, learner: dict, cohort: list[dict] | None = None) -> Report:
    rep = Report()
    for rule, pattern, desc in CONTENT_RULES:
        m = re.search(pattern, text or "", re.I)
        if m:
            rep.block(rule, f"{desc}: “{m.group(0)}”")
    # privacy: never mention other learners
    for other in cohort or []:
        if other["learner_id"] == learner["learner_id"]:
            continue
        if other["email"].lower() in (text or "").lower() or re.search(rf"\b{re.escape(other['name'])}\b", text or ""):
            rep.block("privacy", f"mentions another learner ({other['name']})")
    emails = set(re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text or "")) - {learner["email"], config.SENDER}
    for e in emails:
        if not e.endswith("eda-academy.example"):
            rep.block("privacy", f"contains a third-party email address ({e})")
    if len((text or "").split()) > 260:
        rep.warn("length", "message is long; learners respond better to short messages")
    if learner.get("first_name") and learner["first_name"] not in (text or ""):
        rep.warn("personalisation", "message does not address the learner by name")
    return rep


def check_contact(learner: dict, state: dict, day: int) -> Report:
    rep = Report()
    if learner.get("do_not_contact"):
        rep.block("do_not_contact", "learner has opted out of outreach: route to a human, do not message")
    last = state.get("last_contact_day")
    if last is not None and day - last < config.CONTACT_COOLDOWN_DAYS:
        rep.block("cooldown", f"contacted {day - last} days ago (minimum gap {config.CONTACT_COOLDOWN_DAYS} days)")
    if (state.get("attempts") or 0) >= config.MAX_ATTEMPTS:
        rep.block("max_attempts", f"{state['attempts']} automated attempts already: hand over to a human")
    return rep


def require_approval(intervention: dict):
    if intervention.get("status") != "approved" or not intervention.get("reviewer"):
        raise PermissionError(f"intervention {intervention.get('id')} has not been approved by a human reviewer")
