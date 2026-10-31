"""End-to-end demo: detect → explain → approve → act → measure → re-plan.

    python demo.py            # interactive approval checkpoint
    python demo.py --auto     # non-interactive (auto-approves as "Coach Asha")
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from eda import config, drafting
from eda.diagnoser import CAUSE_LABEL
from eda.planner import CATALOG
from eda.runner import Runner
from eda.tools import ToolBox

B, D, R, G, Y, C, X = "\033[1m", "\033[2m", "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[0m"


def h(title):
    print(f"\n{B}{C}{'═' * 88}\n  {title}\n{'═' * 88}{X}")


def show_trace(run, lid, since_day):
    for e in run.mem.trace(lid):
        if e["day"] < since_day:
            continue
        col = {"plan": C, "replan": Y, "guardrail": R, "approval": G, "outcome": B, "act": D}.get(e["phase"], "")
        print(f"  {col}[{e['phase']:9}]{X} {e['title'][:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    reviewer = "Coach Asha"

    run = Runner()
    run.reset(args.seed)
    print(f"{D}LLM drafting: {drafting.llm_status()} · SMTP sink {config.SMTP_HOST}:{config.SMTP_PORT} · data in {config.DATA_DIR}{X}")

    # ------------------------------------------------------------------ 1. detect
    h(f"DAY {run.day} · Agent cycle 1: monitor all learners")
    summary = run.run_cycle()
    states = sorted(run.mem.all_states(), key=lambda s: -(s["risk"] or 0))
    print(f"  {'id':5} {'name':18} {'risk':>5}  {'status':18} cause")
    for s in states[:22]:
        lr = run.agent._learner(s["learner_id"])
        risk = f"{s['risk']:.2f}" if s["risk"] is not None else "  -  "
        col = R if s["status"] in ("awaiting_approval", "escalated") else (Y if s["status"] in ("watch", "insufficient_data") else "")
        print(f"  {col}{s['learner_id']:5} {lr['name']:18} {risk:>5}  {s['status']:18} {CAUSE_LABEL.get(s['cause'], '')}{X}")
    print(f"\n  {summary}")

    # ------------------------------------------------------------------ 2. explain
    pending = sorted(run.mem.interventions(status="pending_approval"), key=lambda i: i["learner_id"])
    focus = next((i for i in pending if i["type"] == "doubt_session"), pending[0])
    lid = focus["learner_id"]
    st = run.mem.state(lid)
    lr = run.agent._learner(lid)
    h(f"Focus learner {lid} · {lr['name']}: why the agent flagged them")
    print(textwrap.fill(st["explanation"], 100, initial_indent="  ", subsequent_indent="  "))
    print(f"\n  {B}Cause probabilities{X}")
    for c, p in sorted(st["cause_probs"].items(), key=lambda x: -x[1]):
        print(f"    {CAUSE_LABEL[c]:28} {'█' * int(p * 40):40} {p:.0%}")
    print(f"\n  {B}Agent trace (plan → act → observe → re-plan){X}")
    show_trace(run, lid, run.day)

    # ------------------------------------------------------------------ 3. approve (human checkpoint)
    h("HUMAN APPROVAL CHECKPOINT")
    print(f"  Proposed: {B}{focus['rationale']}{X}\n  Subject: {focus['subject']}\n")
    print(textwrap.indent(focus["body"], "  │ "))
    decision, body = "approve", None
    if not args.auto:
        ans = input(f"\n  {B}[a]pprove / [e]dit / [r]eject?{X} ").strip().lower() or "a"
        if ans.startswith("r"):
            decision = "reject"
        elif ans.startswith("e"):
            print("  Enter replacement body, end with a line containing only '.'")
            lines = []
            while (ln := input()) != ".":
                lines.append(ln)
            body = "\n".join(lines)
    res = run.agent.review(focus["id"], decision, reviewer, body=body)
    if res.get("refused"):
        print(f"\n  {R}🛑 Guardrail refused your edit:{X} {[v['detail'] for v in res['violations']]}. Sending the original instead.")
        res = run.agent.review(focus["id"], "approve", reviewer)
    print(f"\n  → {res}")
    for f in sorted(Path(config.OUTBOX_DIR).glob("*.eml"))[-1:]:
        print(f"  {G}✉ delivered over SMTP → {f}{X}")
    if res.get("session_file"):
        print(f"  {G}📅 calendar invite → {res['session_file']}{X}")

    # ------------------------------------------------------------------ guardrail demos
    h("GUARDRAILS: refusal & failure handling")
    fin = next((i for i in sorted(run.mem.interventions(status="pending_approval"), key=lambda i: i["learner_id"])
                if i["type"] == "financial_counsellor_call"), None)
    if fin:
        bad = fin["body"] + "\n\nPlease clear your dues by Friday or we will cancel your enrollment."
        print(f"  1) Reviewer edits {fin['learner_id']}'s message to add: “…or we will cancel your enrollment.”")
        r = run.agent.review(fin["id"], "approve", "Ops Manager", body=bad)
        print(f"     {R}🛑 REFUSED{X}: {[v['detail'] for v in r['violations']]}  (message stays in the queue, unsent)")
    esc = [i for i in run.mem.interventions(status="escalated")]
    for e in esc:
        if "opted out" in (e["rationale"] or ""):
            print(f"  2) {e['learner_id']}: {R}🛑 automated outreach refused{X}: {e['rationale']}")
        else:
            print(f"  2) {e['learner_id']}: {Y}abstained from an automated guess{X} ({e['rationale']}) → human counsellor queue")
    other = next((i for i in sorted(run.mem.interventions(status="pending_approval"), key=lambda i: i["learner_id"])
                  if i["type"] != "financial_counsellor_call"), None)
    if other:
        print(f"  3) Mail server outage while sending #{other['id']} ({other['learner_id']}):")
        good_tools = run.agent.tools
        run.agent.tools = ToolBox(smtp_port=1, retry_backoff=(0.2, 0.4))
        r = run.agent.review(other["id"], "approve", reviewer)
        run.agent.tools = good_tools
        print(f"     send_message retried {Y}3×{X} → {r['status']}: kept in queue, follow-up clock not started; will retry next cycle")

    # approve the rest
    rest = run.mem.interventions(status="pending_approval")
    for i in rest:
        run.agent.review(i["id"], "approve", reviewer)
    print(f"\n  {reviewer} approved the remaining {len(rest)} drafts.")

    # ------------------------------------------------------------------ 4. measure
    run.advance(config.FOLLOWUP_DAYS)
    h(f"DAY {run.day} · Agent cycle 2: measure outcomes, re-plan where needed")
    run.run_cycle()
    print(f"  {'id':5} {'intervention':32} {'before':>6} {'after':>6} {'healthy':>7}  result")
    for iv in run.mem.interventions(status="measured"):
        e = iv["outcome"]["engagement"]
        ok = f"{G}recovered{X}" if iv["success"] else f"{R}not working → re-plan{X}"
        print(f"  {iv['learner_id']:5} {CATALOG[iv['type']]['label']:32} {e['before']['index']:6.2f} {e['after']['index']:6.2f} "
              f"{e['healthy_baseline']['index']:7.2f}  {ok}")
    retried = [e for e in run.mem.trace() if e["title"].startswith("Retrying delivery")]
    if retried:
        print(f"\n  {G}✓ the message that failed during the outage was re-delivered at the start of this cycle{X}")
    h(f"Focus learner {lid}: measured outcome")
    show_trace(run, lid, run.day)

    replans = sorted(run.mem.interventions(status="pending_approval"), key=lambda i: i["learner_id"])
    if replans:
        h("RE-PLAN: first intervention failed, agent proposes a different one")
        for i in replans:
            first = [x for x in run.mem.interventions(i["learner_id"], status="measured")][-1]
            print(f"  {i['learner_id']}: {CATALOG[first['type']]['label']} failed → now proposing {B}{CATALOG[i['type']]['label']}{X}")
        show_trace(run, replans[0]["learner_id"], run.day)
        for i in replans:
            run.agent.review(i["id"], "approve", reviewer)
        run.advance(config.FOLLOWUP_DAYS)
        h(f"DAY {run.day} · Agent cycle 3: measure the re-planned interventions")
        run.run_cycle()
        for i in replans:
            iv = run.mem.intervention(i["id"])
            e = iv["outcome"]["engagement"]
            ok = f"{G}recovered{X}" if iv["success"] else f"{R}still not working → re-plan / human escalation{X}"
            print(f"  {iv['learner_id']:5} {CATALOG[iv['type']]['label']:32} {e['before']['index']:6.2f} → {e['after']['index']:5.2f}  {ok}")

    h("SUMMARY")
    ivs = run.mem.interventions(status="measured")
    print(f"  interventions measured: {len(ivs)}, successful: {sum(i['success'] for i in ivs)}")
    print(f"  learner statuses: {run.agent.summary()['statuses']}")
    print(f"  memory DB: {config.MEMORY_DB}\n  outbox: {config.OUTBOX_DIR} ({len(list(Path(config.OUTBOX_DIR).glob('*.eml')))} emails)"
          f"\n  invites: {config.CALENDAR_DIR}")
    print(f"\n  Explore it visually:  {B}streamlit run app.py{X}")


if __name__ == "__main__":
    main()
