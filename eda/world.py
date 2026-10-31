"""Synthetic learning-platform world.

Generates realistic, seeded learner activity (logins, live-class attendance, quiz
scores, topic progress, payments, support tickets, pulse surveys) into
``platform.db``. Each learner has a hidden archetype (the *true* cause of
disengagement) stored in ``_world_truth``; the agent's SQL tool cannot read it.

The world is also the environment for closed-loop evaluation: when the agent
delivers an intervention, the world re-simulates the following days, and the
learner recovers with a probability that depends on whether the intervention
matched the true cause (``EFFECT``).
"""
from __future__ import annotations

import hashlib
import random
import sqlite3
from dataclasses import dataclass, field

from . import config

TOPICS = [
    "Python Basics", "Control Flow", "Functions", "Recursion", "Data Structures",
    "SQL Joins", "Statistics", "Linear Regression", "Classification",
    "Neural Networks", "Capstone",
]

CAUSES = ["stuck_on_topic", "low_scores", "financial", "motivation"]
INTERVENTIONS = ["doubt_session", "easier_plan", "counsellor_call", "financial_counsellor_call", "motivation_nudge"]

# P(learner recovers | true cause, intervention). Ground truth of the simulator;
# the agent starts with its own (similar but not identical) prior in planner.py.
EFFECT = {
    "stuck_on_topic": {"doubt_session": .88, "easier_plan": .45, "counsellor_call": .25, "financial_counsellor_call": .02, "motivation_nudge": .12},
    "low_scores":     {"doubt_session": .40, "easier_plan": .85, "counsellor_call": .30, "financial_counsellor_call": .02, "motivation_nudge": .10},
    "financial":      {"doubt_session": .03, "easier_plan": .10, "counsellor_call": .45, "financial_counsellor_call": .88, "motivation_nudge": .05},
    "motivation":     {"doubt_session": .12, "easier_plan": .30, "counsellor_call": .82, "financial_counsellor_call": .05, "motivation_nudge": .50},
}

ARCH_CAUSE = {"stuck": "stuck_on_topic", "low_scores": "low_scores", "financial": "financial",
              "motivation": "motivation", "mixed": "motivation", "dnc": "motivation"}

# archetype -> (login drop, attendance drop, ramp days)
ARCH_PARAMS = {
    "stuck":      (.55, .50, 8),
    "low_scores": (.50, .45, 20),
    "financial":  (.92, .95, 3),
    "motivation": (.80, .85, 18),
    "mixed":      (.70, .70, 18),
    "dnc":        (.80, .85, 18),
}

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha", "Arjun", "Sara",
         "Neel", "Priya", "Dev", "Tara", "Kiran", "Zoya", "Aditya", "Riya", "Omar", "Leela",
         "Sam", "Nisha", "Yash", "Fatima", "Ishaan", "Mira", "Raghav", "Aisha", "Varun", "Elena",
         "Kunal", "Sneha", "Hugo", "Pooja", "Farhan", "Lina", "Manav", "Asha", "Tomas", "Gauri"]
LAST = ["Sharma", "Iyer", "Khan", "Patel", "Das", "Nair", "Rao", "Gupta", "Menon", "Joshi",
        "Singh", "Kulkarni", "Reddy", "Bose", "Fernandes", "Mehta", "Pillai", "Chopra", "Shah", "Verma"]


def _rng(*parts) -> random.Random:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


@dataclass
class LearnerSpec:
    learner_id: str
    name: str
    email: str
    archetype: str
    enrolled_day: int = 0
    plan: str = "upfront"
    do_not_contact: int = 0
    onset: int = 999
    due_day: int = 15
    ability: float = 78.0
    base_login: float = .86
    vac_start: int = -1
    seed: int = 0
    stuck_topic: str | None = None


@dataclass
class SimOutput:
    logins: list = field(default_factory=list)
    attendance: list = field(default_factory=list)
    quizzes: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    payments: list = field(default_factory=list)
    tickets: list = field(default_factory=list)
    surveys: list = field(default_factory=list)
    stuck_topic: str | None = None


COHORT_MIX = (["healthy"] * 18 + ["stuck"] * 5 + ["low_scores"] * 4 + ["financial"] * 4 +
              ["motivation"] * 4 + ["mixed", "vacation", "vacation", "new", "dnc"])


def make_cohort(seed: int) -> list[LearnerSpec]:
    rng = random.Random(seed)
    archs = COHORT_MIX[:]
    rng.shuffle(archs)
    specs = []
    names = rng.sample([(f, l) for f in FIRST for l in LAST], len(archs))
    for i, arch in enumerate(archs):
        first, last = names[i]
        s = LearnerSpec(
            learner_id=f"L{i + 1:03d}", name=f"{first} {last}",
            email=f"{first.lower()}.{last.lower()}@learners.example", archetype=arch, seed=seed,
            ability=rng.gauss(79, 4), base_login=rng.uniform(.78, .93),
            plan="emi" if (arch == "financial" or rng.random() < .4) else "upfront",
        )
        if arch == "stuck":
            s.onset = rng.randint(12, 17)
        elif arch == "low_scores":
            s.onset, s.ability = rng.randint(3, 6), rng.gauss(72, 3)
        elif arch == "financial":
            s.due_day = rng.randint(14, 21)
            s.onset = s.due_day + 1
        elif arch in ("motivation", "dnc"):
            s.onset = rng.randint(7, 11)
        elif arch == "mixed":
            s.onset, s.ability = 8, rng.gauss(74, 2)
        elif arch == "vacation":
            s.vac_start = rng.randint(19, 21)
        elif arch == "new":
            s.enrolled_day = config.DETECTION_DAY - 5
        if arch == "dnc":
            s.do_not_contact = 1
        specs.append(s)
    return specs


def _recovery(spec: LearnerSpec, day: int, delivered: list[tuple[str, int]]) -> float:
    """How much of the disengagement has been reversed on `day` by delivered interventions."""
    cause = ARCH_CAUSE.get(spec.archetype)
    if not cause:
        return 0.0
    r = 0.0
    for itype, sent_day in delivered:
        if day <= sent_day:
            continue
        eff = EFFECT[cause].get(itype, 0.0)
        ramp = min(1.0, (day - sent_day) / 4)
        success = _rng(spec.seed, spec.learner_id, itype, sent_day, "outcome").random() < eff
        r = max(r, (0.92 if success else 0.25 * eff) * ramp)
    return r


def simulate(spec: LearnerSpec, upto_day: int, delivered: list[tuple[str, int]] | None = None) -> SimOutput:
    """Deterministically simulate one learner for days [enrolled_day, upto_day)."""
    delivered = delivered or []
    out = SimOutput()
    arch = spec.archetype
    drop_login, drop_att, ramp = ARCH_PARAMS.get(arch, (0, 0, 1))
    topic_idx, active_on_topic, attempts = 0, 0, {}
    stuck_topic = None
    fin_recovered_day = None

    for day in range(spec.enrolled_day, upto_day):
        rng = _rng(spec.seed, spec.learner_id, day)
        sev = min(1.0, max(0.0, (day - spec.onset) / ramp)) if day >= spec.onset else 0.0
        rec = _recovery(spec, day, delivered)
        s = sev * (1 - rec)
        if arch == "stuck" and day >= spec.onset and stuck_topic is None:
            stuck_topic = TOPICS[topic_idx]
        if arch == "financial" and rec > .5 and fin_recovered_day is None:
            fin_recovered_day = day
        on_vacation = spec.vac_start <= day < spec.vac_start + 5
        topic = TOPICS[topic_idx]
        is_stuck_now = arch == "stuck" and topic == stuck_topic and rec < .5

        # --- login & study time
        p_login = 0.0 if on_vacation else spec.base_login * (1 - s * drop_login)
        if rng.random() < p_login:
            minutes = max(5, int(rng.gauss(55, 15) * (1 - (.3 if arch == "stuck" else .6) * s)))
            out.logins.append((spec.learner_id, day, minutes, topic))
            p = out.progress.setdefault(topic, {"started_day": day, "completed_day": None, "time": 0, "rewatches": 0})
            p["time"] += minutes
            p["rewatches"] += _poisson(rng, 0.4 + 2.5 * s if is_stuck_now else 0.3)
            active_on_topic += 1
            # --- quizzes
            if rng.random() < (.45 + .35 * s if is_stuck_now else .45):
                mu = spec.ability
                if arch == "stuck" and topic == stuck_topic:
                    mu -= 42 * s
                elif arch == "low_scores":
                    mu -= 8 + 26 * s
                elif arch == "mixed":
                    mu -= 14 * s
                elif arch in ("motivation", "dnc"):
                    mu -= 6 * s
                elif arch == "financial":
                    mu -= 3 * s
                attempts[topic] = attempts.get(topic, 0) + 1
                score = round(min(100, max(5, rng.gauss(mu, 7))), 1)
                out.quizzes.append((spec.learner_id, day, topic, score, attempts[topic]))
            # --- topic advancement (stuck learners don't advance)
            if active_on_topic >= 4 and not is_stuck_now and topic_idx < len(TOPICS) - 1:
                out.progress[topic]["completed_day"] = day
                topic_idx, active_on_topic = topic_idx + 1, 0

        # --- live classes on Tue/Fri (day % 7 in 1, 4)
        if day % 7 in (1, 4):
            attended = 0 if on_vacation else int(rng.random() < 0.92 * (1 - s * drop_att))
            out.attendance.append((spec.learner_id, day, attended))

        # --- weekly pulse survey
        if day % 7 == 0 and day > spec.enrolled_day and rng.random() < .8:
            out.surveys.append(_survey(spec, day, s, rng))

    # --- payments
    if spec.plan == "upfront":
        out.payments.append((spec.learner_id, spec.enrolled_day, spec.enrolled_day, 59999, "paid"))
    else:
        for due in (spec.due_day, spec.due_day + 30):
            if due >= upto_day + 3:
                continue
            if arch == "financial" and due == spec.due_day:
                if fin_recovered_day is not None:
                    out.payments.append((spec.learner_id, due, fin_recovered_day, 4999, "restructured"))
                else:
                    out.payments.append((spec.learner_id, due, None, 4999, "failed" if upto_day - due < 3 else "overdue"))
            elif due < upto_day:
                late = _rng(spec.seed, spec.learner_id, due, "pay").randint(0, 2)
                out.payments.append((spec.learner_id, due, due + late, 4999, "paid"))
            else:
                out.payments.append((spec.learner_id, due, None, 4999, "upcoming"))

    out.tickets = [t for t in _tickets(spec, stuck_topic) if t[1] < upto_day]
    out.stuck_topic = stuck_topic
    return out


def _poisson(rng: random.Random, lam: float) -> int:
    k, p, L = 0, 1.0, pow(2.718281828, -lam)
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def _survey(spec, day, s, rng):
    arch = spec.archetype
    comments = {
        "motivation": ["Too busy with work lately", "Not sure this course is worth it anymore",
                       "Feeling bored by the pace", "Honestly losing interest"],
        "mixed": ["Feeling a bit unmotivated", "Hard to find the energy after work", "Not enjoying it as much"],
        "dnc": ["Busy", "Losing interest"],
        "stuck": ["Some topics are really confusing"],
        "low_scores": ["Feeling lost in the assessments", "Quizzes are harder than expected"],
        "financial": ["Worried about fees", "Stressful month"],
    }
    if arch in ("motivation", "mixed", "dnc"):
        m = 4.2 - 2.8 * s
    elif arch in ("stuck", "low_scores"):
        m = 4.0 - 1.0 * s
    elif arch == "financial":
        m = 4.0 - 1.2 * s
    else:
        m = 4.3
    score = int(min(5, max(1, round(rng.gauss(m, .5)))))
    comment = rng.choice(comments[arch]) if (arch in comments and s > .3) else rng.choice(["", "Good so far", "Enjoying the projects", ""])
    return (spec.learner_id, day, score, comment)


def _tickets(spec, stuck_topic):
    arch, lid = spec.archetype, spec.learner_id
    rng = _rng(spec.seed, lid, "tickets")
    t = []
    if arch == "stuck" and stuck_topic:
        t += [(lid, spec.onset + 3, "content", f"I've watched the {stuck_topic} videos three times and still can't solve the practice problems."),
              (lid, spec.onset + 7, "content", f"Is there someone who can explain {stuck_topic}? I keep failing the quiz.")]
    elif arch == "low_scores":
        t.append((lid, 14, "assessment", "My quiz scores keep dropping across modules, not sure what I'm doing wrong."))
    elif arch == "financial":
        t.append((lid, spec.due_day + 1, "billing", rng.choice([
            "My EMI payment failed. I lost my job last month and money is tight right now.",
            "Can I pause my payments? I'm facing some financial difficulty this month."])))
    elif arch == "vacation":
        t.append((lid, spec.vac_start - 1, "general", f"Travelling for a family wedding from day {spec.vac_start}, back in 5 days."))
    elif arch == "mixed":
        t.append((lid, 18, "assessment", "I'm finding the assignments harder to get through lately."))
    elif arch == "healthy" and rng.random() < .3:
        t.append((lid, rng.randint(3, 25), "technical", "Video player keeps buffering on mobile."))
    return t


# ----------------------------------------------------------------------------- persistence

SCHEMA = """
CREATE TABLE IF NOT EXISTS learners(learner_id TEXT PRIMARY KEY, name TEXT, email TEXT, cohort TEXT,
    enrolled_day INT, plan TEXT, do_not_contact INT, preferred_channel TEXT);
CREATE TABLE IF NOT EXISTS logins(learner_id TEXT, day INT, minutes INT, topic TEXT);
CREATE TABLE IF NOT EXISTS attendance(learner_id TEXT, day INT, attended INT);
CREATE TABLE IF NOT EXISTS quiz_scores(learner_id TEXT, day INT, topic TEXT, score REAL, attempt INT);
CREATE TABLE IF NOT EXISTS topic_progress(learner_id TEXT, topic TEXT, started_day INT, completed_day INT,
    time_spent_min INT, video_rewatches INT);
CREATE TABLE IF NOT EXISTS payments(learner_id TEXT, due_day INT, paid_day INT, amount INT, status TEXT);
CREATE TABLE IF NOT EXISTS support_tickets(learner_id TEXT, day INT, category TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS surveys(learner_id TEXT, day INT, motivation INT, comment TEXT);
CREATE TABLE IF NOT EXISTS _world_truth(learner_id TEXT PRIMARY KEY, archetype TEXT, cause TEXT, onset_day INT,
    stuck_topic TEXT, ability REAL, base_login REAL, due_day INT, vac_start INT, seed INT);
CREATE TABLE IF NOT EXISTS _world_meta(key TEXT PRIMARY KEY, value TEXT);
"""
ACTIVITY_TABLES = ["logins", "attendance", "quiz_scores", "topic_progress", "payments", "support_tickets", "surveys"]


class World:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.PLATFORM_DB)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def reset(self, seed: int = 7, day: int = config.DETECTION_DAY):
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            for tbl in ["learners", *ACTIVITY_TABLES, "_world_truth", "_world_meta"]:
                c.execute(f"DROP TABLE IF EXISTS {tbl}")
            c.executescript(SCHEMA)
            for s in make_cohort(seed):
                c.execute("INSERT INTO learners VALUES(?,?,?,?,?,?,?,?)",
                          (s.learner_id, s.name, s.email, "DS-2026-Oct", s.enrolled_day, s.plan, s.do_not_contact, "email"))
                c.execute("INSERT INTO _world_truth VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (s.learner_id, s.archetype, ARCH_CAUSE.get(s.archetype), s.onset, None, s.ability,
                           s.base_login, s.due_day, s.vac_start, seed))
            c.execute("INSERT INTO _world_meta VALUES('seed', ?)", (str(seed),))
        self.advance_to(day, {})

    def specs(self) -> list[LearnerSpec]:
        with self._conn() as c:
            rows = c.execute("""SELECT l.learner_id, l.name, l.email, t.archetype, l.enrolled_day, l.plan,
                l.do_not_contact, t.onset_day, t.due_day, t.ability, t.base_login, t.vac_start, t.seed
                FROM learners l JOIN _world_truth t USING(learner_id) ORDER BY l.learner_id""").fetchall()
        return [LearnerSpec(*r) for r in rows]

    @property
    def clock(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT value FROM _world_meta WHERE key='clock'").fetchone()
        return int(row[0]) if row else 0

    def advance_to(self, day: int, delivered: dict[str, list[tuple[str, int]]]):
        """Regenerate all activity up to `day` given interventions delivered so far.

        Simulation is deterministic per (learner, day), and interventions only act
        on days after they were sent, so history before an intervention never changes.
        """
        specs = self.specs()
        with self._conn() as c:
            for tbl in ACTIVITY_TABLES:
                c.execute(f"DELETE FROM {tbl}")
            for s in specs:
                o = simulate(s, day, delivered.get(s.learner_id, []))
                c.executemany("INSERT INTO logins VALUES(?,?,?,?)", o.logins)
                c.executemany("INSERT INTO attendance VALUES(?,?,?)", o.attendance)
                c.executemany("INSERT INTO quiz_scores VALUES(?,?,?,?,?)", o.quizzes)
                c.executemany("INSERT INTO topic_progress VALUES(?,?,?,?,?,?)",
                              [(s.learner_id, t, p["started_day"], p["completed_day"], p["time"], p["rewatches"])
                               for t, p in o.progress.items()])
                c.executemany("INSERT INTO payments VALUES(?,?,?,?,?)", o.payments)
                c.executemany("INSERT INTO support_tickets VALUES(?,?,?,?)", o.tickets)
                c.executemany("INSERT INTO surveys VALUES(?,?,?,?)", o.surveys)
                if o.stuck_topic:
                    c.execute("UPDATE _world_truth SET stuck_topic=? WHERE learner_id=?", (o.stuck_topic, s.learner_id))
            c.execute("INSERT OR REPLACE INTO _world_meta VALUES('clock', ?)", (str(day),))

    def truth(self) -> dict[str, dict]:
        with self._conn() as c:
            c.row_factory = sqlite3.Row
            return {r["learner_id"]: dict(r) for r in c.execute("SELECT * FROM _world_truth")}
