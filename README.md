# Rekindle: Early-Disengagement Detection and Re-engagement Agent

On online learning platforms, disengagement is usually noticed weeks too late, typically once a learner has gone silent.
This agent watches login, attendance and score **trends**, flags risk early, infers the **probable cause**
(stuck on a topic, low scores, money, motivation), picks a **matching intervention** (doubt session, easier plan,
counsellor call, …), **drafts it for human approval**, sends it, and **measures whether it worked**. If it didn't, the
agent re-plans.

```
             ┌──────────────────────── memory (SQLite, persists across cycles) ────────────────────────┐
             │ learner_state · interventions · measured outcomes · learned effectiveness · full trace  │
             └───────────────▲───────────────────────────────▲────────────────────────────▲────────────┘
PLAN ──► ACT ───────────────►│ OBSERVE ─────────────────────►│ RE-PLAN ──────────────────►│
goal per learner   sql_query (read-only LMS DB)     risk, drivers, evidence    false alarm → keep watching
from memory state  run_python sandbox (analytics)   tool failures              opted-out → human queue
                   search_resources (TF-IDF)        guardrail verdicts         sandbox down → degraded mode
                   send_message (SMTP + Slack)      measured outcome           intervention failed → Bayesian
                   schedule_session (.ics)                                       update, re-diagnose, new plan
                            │
                   ⏸ HUMAN APPROVAL CHECKPOINT (named reviewer, edits re-checked by policy)
```

## Quick start

```bash
pip install -r requirements.txt          # pandas, scikit-learn, streamlit, altair
python demo.py                           # scripted end-to-end story; you are the approver ([a]pprove/[e]dit/[r]eject)
python demo.py --auto                    # same, non-interactive
streamlit run app.py                     # dashboard: risk radar, approval queue, learner deep-dive, outcomes, trace
python evaluate.py                       # evaluation set + metrics → EVAL_REPORT.md (~35 s)
```

**LLM: NVIDIA Nemotron 3 Ultra** (`nvidia/nemotron-3-ultra-550b-a55b`, NVIDIA's OpenAI-compatible API) writes the
counsellor case notes and personalises the learner drafts. Put the key in `.env` (gitignored, never in code):

```bash
echo "NVIDIA_API_KEY=nvapi-..." > .env
```

Settings (env vars): `EDA_LLM_MODEL`, `EDA_LLM_BASE_URL`, `EDA_LLM_RPM` (client-side rate cap, default 35/min for
the free tier), `EDA_LLM_WORKERS` (learners processed in parallel, default 4), `EDA_LLM_THINKING=on` (enable the
model's reasoning; off by default because it adds latency to short rewrites), `EDA_USE_LLM=off`.
Without a key, or when a call fails (timeout, 429 after back-off, truncated output), that draft falls back to a vetted
template and the trace records why (e.g. `template (HTTP 429)`). LLM output is never trusted blindly: it goes through
the same content guardrails, and a draft that fails is replaced by the template. The drafting prompt gets no internal
tracking evidence, so the model can't tell a learner what we observed about them. A full demo with the LLM takes ~3 min
(mostly the rate cap). `evaluate.py` runs with the LLM off by default so metrics are deterministic and free.

### Adding a learner and emailing them for real

```bash
python add_learner.py --id L042 --name "Harsh Satbhai" --email you@gmail.com --onset 14 --seed 592   # risk ≈ 0.78
```

Risk is never typed in: the learner gets a simulated history and the agent scores them on the next cycle.
Simulated learners are only ever "emailed" into the local outbox. To deliver a real email (with the `.ics`
invite attached) to specific addresses, add a Gmail [app password](https://myaccount.google.com/apppasswords) to `.env`:

```bash
EDA_RELAY_HOST=smtp.gmail.com
EDA_RELAY_PORT=587
EDA_RELAY_USER=you@gmail.com
EDA_RELAY_PASSWORD=xxxxxxxxxxxxxxxx      # 16-char app password, not your Google password
EDA_REAL_RECIPIENTS=you@gmail.com        # comma-separated allowlist; nobody else gets real mail
```

The human approval checkpoint still applies, and the local copy shows as 📨 REAL in the dashboard outbox.

## What the demo shows (the required demo)

| Required | Where |
|---|---|
| **At-risk learner detected** | Day 28 cycle: e.g. *L004 Ananya Nair*, risk 0.71 (quiz avg 44% vs 69%, active 29% of days vs 79%) |
| **Cause explained** | "Stuck on a topic (54%)": 3 failed attempts on *Functions*, 15 days on the module (typical ~5), 7 video rewatches, ticket "Is there someone who can explain Functions?". Alternatives with probabilities. |
| **Approved intervention** | 1:1 doubt-clearing session drafted with 3 resources found by search; human approves, then an email goes out over SMTP and a `.ics` invite is created |
| **Measured outcome** | 10 days later: engagement 0.39 → 0.85 vs healthy baseline 0.69 → *recovered*. Another learner's easier plan fails (0.51 → 0.40), the agent re-plans to a doubt session, and it then recovers (0.40 → 0.87) |

## Minimum bar → implementation

| Requirement | Implementation |
|---|---|
| **Agentic loop** | [eda/agent.py](eda/agent.py): per-learner `plan → act → observe → re-plan` executor. Each action returns an observation that may replace the rest of the plan (explained absence → stop; opted out → escalate; LLM draft fails policy → template; outcome failed → re-diagnose + new intervention). Every step is logged to the trace. |
| **≥3 real tools** | [eda/tools.py](eda/tools.py): **(1)** `sql_query`, read-only SQLite with a table allowlist enforced by SQLite's authorizer; **(2)** `run_python`, code execution in an isolated interpreter (`-I`, empty env, AST import allowlist, CPU/file rlimits, timeout) that runs [eda/analytics.py](eda/analytics.py); **(3)** `search_resources`, a TF-IDF search engine over a 55-item resource catalog; **(4)** `send_message`, real SMTP via `smtplib` with retry/backoff, plus an optional Slack webhook (`EDA_SLACK_WEBHOOK_URL`); **(5)** `schedule_session`, RFC 5545 calendar invites. A local SMTP server ([eda/mailsink.py](eda/mailsink.py)) captures mail in `data/outbox/`; point `EDA_SMTP_HOST/PORT` at a real relay for production. |
| **Memory** | [eda/memory.py](eda/memory.py): `data/agent_memory.db` stores learner status, risk, cause beliefs, intervention history, outcomes, **learned intervention effectiveness** (a Beta posterior used by the planner) and the full trace. Kill the process and the next cycle picks up pending follow-ups. |
| **Evaluation (≥10 cases + metric)** | [evaluate.py](evaluate.py): 120 labelled learner cases (3 cohorts × 40) + 19 guardrail/fault-injection cases. Report: [EVAL_REPORT.md](EVAL_REPORT.md). |
| **Guardrails** | [eda/guardrails.py](eda/guardrails.py): **human approval checkpoint** (`Agent.review`; the send path calls `require_approval` again as a hard gate). **Refusals:** opted-out learner (no outreach, routed to a human), contact cooldown, max attempts, and a content policy that blocks threats, shaming, risk-label disclosure, mentions of tracked behaviour, other learners' data and job guarantees. A reviewer edit like "…or we will cancel your enrollment" is **refused**. **Failure handling:** SMTP outage (retry ×3 → `send_failed` → re-delivered next cycle without starting the follow-up clock), sandbox crash (degraded in-process analytics), too little history (abstain), cause unclear (escalate instead of guessing). |
| **Data** | [eda/world.py](eda/world.py): seeded synthetic LMS (logins, live classes, quizzes, topic progress, EMI payments, support tickets, pulse surveys) with hidden archetypes. It also acts as the closed-loop environment: learners respond to delivered interventions with a probability that depends on whether the intervention matches the true cause. |

## Results ([EVAL_REPORT.md](EVAL_REPORT.md), seeds 7/11/23)

| Metric | Value |
|---|---|
| Detection precision / recall / F1 at day 28 | 0.98 / 0.91 / 0.95 |
| Cause-diagnosis accuracy (true positives) | 1.00 (n=52) |
| First intervention matches true cause | 0.96 (n=53) |
| Median early-warning lead time vs "7 days inactive" rule | **15 days** (agent flags a median of 9 days after onset; the naive rule takes 22.5, and never fires for 17 of 57) |
| Re-engaged at day 58 (engagement index ≥ 0.55): agent vs no intervention | **88% vs 7%** |
| Interventions that worked (agent's own measurement) | 85% of 65; 14 re-plans after failures |
| Guardrail & failure-handling cases | **19/19** |

Held-out seeds 101/202/303/404 (never used for tuning): F1 0.98, cause accuracy 0.99, re-engaged 91% vs 9%.

**Read these numbers honestly.** The data is synthetic, and I designed the signals, so diagnosis is easier than in real
life (that's why cause accuracy is ~1.0). Treat the eval as proof that the loop, guardrails and measurement work end to end, not as
a claim about real-world accuracy. Where the naive rule never fires, lead time is counted to the horizon (a lower bound).
The intervention effect sizes in the simulator are assumptions. The agent starts from its own similar but different
prior ([eda/planner.py](eda/planner.py)) and updates it from measured outcomes.

## How it decides

- **Risk** ([eda/analytics.py](eda/analytics.py)): drops vs the learner's own baseline (best recent weeks, or onboarding
  if their baseline has drifted down) in login rate, attendance and quiz scores, plus inactivity, the weekly trend, repeated
  failures on one topic, and low absolute activity. Flag at ≥ 0.21. The output lists the top drivers.
- **Cause** ([eda/diagnoser.py](eda/diagnoser.py)): interpretable evidence scores from 6 SQL queries (quiz failures
  by topic, dwell time and rewatches, payments, tickets, surveys, activity around the EMI due date), normalised into
  probabilities. Failed interventions update them: P(c | i failed) ∝ P(c)·(1 − eff(c,i)).
  An absence the learner explained, followed by their return, is recognised as a false alarm.
- **Intervention** ([eda/planner.py](eda/planner.py)): maximise Σ P(c)·eff(c,i) − λ·cost(i) over interventions not yet tried.
  If nothing is likely to work, or 3 attempts have failed, escalate to a human.
- **Outcome**: engagement index (0.4 logins + 0.3 attendance + 0.3 scores) for the 7 days before the intervention vs days 3–10
  after it. Success means closing ≥ 50% of the gap to the learner's healthy baseline.

## Layout

```
eda/world.py        synthetic platform + simulator (hidden ground truth in _world_truth)
eda/analytics.py    risk & engagement maths (runs inside the code sandbox)
eda/tools.py        the 5 tools + ToolBox dispatcher (timing, retries, failure capture)
eda/diagnoser.py    evidence gathering + cause inference
eda/planner.py      intervention catalogue + expected-utility choice with learned effectiveness
eda/drafting.py     Nemotron / template drafting and case notes (rate-limited HTTP client)
eda/guardrails.py   contact policy, content policy, approval gate
eda/memory.py       persistent memory
eda/agent.py        the loop
eda/runner.py       wires world + agent + mail sink
demo.py · app.py · evaluate.py
```

Moving to real data (not built yet): the agent only reads the eight tables through `sql_query`. Loading a real
export into the same schema in `$EDA_DATA_DIR/platform.db` (e.g. from the Open University **OULAD** dataset:
`studentVle` → logins, `studentAssessment` → quiz_scores) and skipping `World` should be enough. Outcomes would then come
from real behaviour instead of the simulator.
