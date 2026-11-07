"""The Early-Disengagement Detection and Re-engagement Agent.

Each cycle (one simulated day) the agent works through every learner with an
explicit loop:

    PLAN     pick a goal and an initial action list from the learner's memory state
    ACT      execute the next action, calling tools (SQL, sandboxed analytics, search, ...)
    OBSERVE  record what came back (including tool failures and guardrail refusals)
    RE-PLAN  an observation can replace the rest of the plan: a sandbox failure leads to
             degraded analysis, an explained absence means keep monitoring, an opted-out
             learner is escalated to a human, a failed intervention goes back to diagnosis
             with updated beliefs, ...

Outbound actions stop at a human approval checkpoint (Agent.review) and are only
executed after a named reviewer approves.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import analytics, config, drafting, guardrails, planner
from .diagnoser import CAUSE_LABEL, diagnose, gather_evidence
from .memory import Memory
from .tools import ToolBox

ACTIVE_STATUSES = ("pending_approval", "approved", "sent", "send_failed")


@dataclass
class Obs:
    summary: str
    ok: bool = True
    data: object = None
    next: list | None = None      # replace the remaining plan
    replan: str | None = None     # why the plan changed (logged as a re-plan step)
    phase: str = "observe"


@dataclass
class Ctx:
    cycle: int
    day: int
    learner: dict
    state: dict
    activity: dict = field(default_factory=dict)
    risk: dict | None = None
    evidence: dict | None = None
    diag: dict | None = None
    choice: dict | None = None
    resources: list = field(default_factory=list)
    draft: dict | None = None
    guard: dict | None = None
    escalation: str | None = None
    intervention: dict | None = None


class Agent:
    def __init__(self, memory: Memory | None = None, tools: ToolBox | None = None, smtp_port: int | None = None,
                 sandbox: bool = True, verbose: bool = False):
        self.mem = memory or Memory()
        self.tools = tools or ToolBox(smtp_port=smtp_port)
        self.sandbox = sandbox
        self.verbose = verbose
        self._cohort = None

    # ================================================================== public API
    def run_cycle(self, day: int) -> dict:
        cycle = self.mem.next_cycle()
        self.mem.set_meta("day", day)
        self.mem.set_meta("cycle_running", True)   # visible to the UI while we work
        self._cohort = None  # re-read the roster each cycle so newly enrolled learners are picked up
        try:
            learners = self._learners()
            retries = self.retry_failed_sends(day, cycle)
            due = [i for i in self.mem.interventions(status="sent") if i["followup_day"] <= day]
            self._log(cycle, day, None, "plan", f"Cycle {cycle} (day {day}): monitor {len(learners)} learners, "
                      f"measure {len(due)} due follow-ups, retry {retries} failed sends")
            # learners are independent; with a remote LLM in the loop, process them in parallel to hide latency
            workers = config.LLM_WORKERS if drafting.llm_status().startswith("Nemotron") else 1
            if workers > 1:
                with ThreadPoolExecutor(workers) as pool:
                    list(pool.map(lambda lr: self._episode(cycle, day, lr), learners))
            else:
                for lr in learners:
                    self._episode(cycle, day, lr)
            summary = self.summary()
            self._log(cycle, day, None, "observe", "Cycle complete", summary)
            return summary
        finally:
            self.mem.set_meta("cycle_running", False)

    def progress(self) -> dict:
        """Cheap snapshot the UI can poll while a cycle runs (no heavy work)."""
        states = self.mem.all_states()
        ivs = self.mem.interventions()
        drafted_this_cycle = [i for i in ivs if i["status"] == "pending_approval"]
        at_risk = [s for s in states if s.get("status") in ("awaiting_approval", "at_risk", "escalated")]
        return {
            "running": bool(self.mem.get_meta("cycle_running", False)),
            "cycle": self.mem.get_meta("cycle", 0),
            "day": self.mem.get_meta("day", config.DETECTION_DAY),
            "learners_seen": len(states),
            "at_risk": len(at_risk),
            "pending_approval": len(drafted_this_cycle),
            "measured": sum(1 for i in ivs if i["status"] == "measured"),
            "escalated": sum(1 for i in ivs if i["status"] == "escalated"),
        }

    def review(self, iid: int, decision: str, reviewer: str, body: str | None = None, subject: str | None = None,
               note: str | None = None, day: int | None = None) -> dict:
        """Human approval checkpoint. decision: 'approve' | 'reject'."""
        day = self.mem.get_meta("day", config.DETECTION_DAY) if day is None else day
        cycle = self.mem.get_meta("cycle", 0)
        iv = self.mem.intervention(iid)
        if not iv or iv["status"] != "pending_approval":
            return {"ok": False, "error": f"intervention {iid} is not awaiting approval"}
        lid = iv["learner_id"]
        if not reviewer or not reviewer.strip():
            return {"ok": False, "error": "a named reviewer is required"}
        if decision == "reject":
            self.mem.update_intervention(iid, status="rejected", reviewer=reviewer, reviewer_note=note, decided_day=day)
            self.mem.save_state(lid, status="monitoring")
            self._log(cycle, day, lid, "approval", f"❌ Rejected by {reviewer}: {iv['type']}", {"note": note})
            return {"ok": True, "status": "rejected"}

        learner = self._learner(lid)
        final_body, final_subject = body or iv["body"], subject or iv["subject"]
        content = guardrails.check_content(final_body + "\n" + final_subject, learner, self._learners())
        contact = guardrails.check_contact(learner, self.mem.state(lid), day)
        if not (content.allowed and contact.allowed):
            viol = content.violations + contact.violations
            self._log(cycle, day, lid, "guardrail", f"🛑 Refused to send reviewer-approved message for {lid}",
                      {"reviewer": reviewer, "violations": viol})
            return {"ok": False, "refused": True, "violations": viol}
        edited = final_body != iv["body"] or final_subject != iv["subject"]
        self.mem.update_intervention(iid, status="approved", reviewer=reviewer, reviewer_note=note, decided_day=day,
                                     body=final_body, subject=final_subject, guardrail=content.as_dict())
        self._log(cycle, day, lid, "approval", f"✅ Approved by {reviewer}{' (edited)' if edited else ''}: "
                  f"{planner.CATALOG[iv['type']]['label']}", {"note": note})
        return self._execute(iid, day, cycle)

    def retry_failed_sends(self, day: int, cycle: int) -> int:
        failed = self.mem.interventions(status="send_failed")
        for iv in failed:
            self.mem.update_intervention(iv["id"], status="approved")
            self._log(cycle, day, iv["learner_id"], "replan", f"Retrying delivery of intervention #{iv['id']}")
            self._execute(iv["id"], day, cycle)
        return len(failed)

    def summary(self) -> dict:
        states = self.mem.all_states()
        by = {}
        for s in states:
            by[s["status"]] = by.get(s["status"], 0) + 1
        ivs = self.mem.interventions()
        measured = [i for i in ivs if i["status"] == "measured"]
        return {"statuses": by, "pending_approval": sum(i["status"] == "pending_approval" for i in ivs),
                "sent": sum(i["status"] in ("sent", "measured") for i in ivs), "measured": len(measured),
                "successful": sum(bool(i["success"]) for i in measured)}

    # ================================================================== the loop
    def _episode(self, cycle, day, learner):
        lid = learner["learner_id"]
        state = self.mem.state(lid)
        ctx = Ctx(cycle, day, learner, state)
        plan, goal = self._initial_plan(state, day, lid)
        if not plan:
            return None
        self._log(cycle, day, lid, "plan", f"Goal: {goal} · plan: {' → '.join(plan)}")
        steps = 0
        while plan and steps < 30:
            action = plan.pop(0)
            steps += 1
            try:
                obs = getattr(self, f"_a_{action}")(ctx)
            except Exception as e:  # never let one learner crash the cycle
                obs = Obs(f"{action} crashed: {type(e).__name__}: {e}", ok=False, next=["escalate"],
                          replan="unexpected error, handing the case to a human")
                ctx.escalation = obs.summary
            self._log(cycle, day, lid, obs.phase, f"{action}: {obs.summary}", obs.data)
            if obs.next is not None:
                if obs.replan:
                    self._log(cycle, day, lid, "replan", obs.replan, {"new_plan": obs.next})
                plan = list(obs.next)
        return ctx

    def _initial_plan(self, st, day, lid):
        status = st.get("status", "monitoring")
        active = [i for i in self.mem.interventions(lid, status=ACTIVE_STATUSES)]
        if any(i["status"] == "pending_approval" for i in active):
            return [], None
        sent = [i for i in active if i["status"] == "sent"]
        if sent:
            if day >= sent[-1]["followup_day"]:
                return ["fetch_activity", "measure_outcome"], f"measure whether '{sent[-1]['type']}' worked"
            return [], None
        if status == "escalated":
            return ["fetch_activity", "compute_risk"], "monitor (case is with a human)"
        return ["fetch_activity", "compute_risk"], "detect disengagement early"

    # ================================================================== actions
    def _a_fetch_activity(self, c: Ctx):
        lid, out = c.learner["learner_id"], {}
        queries = {
            "logins": "SELECT day, minutes FROM logins WHERE learner_id=? AND day < ? ORDER BY day",
            "attendance": "SELECT day, attended FROM attendance WHERE learner_id=? AND day < ? ORDER BY day",
            "quizzes": "SELECT day, topic, score, attempt FROM quiz_scores WHERE learner_id=? AND day < ? ORDER BY day",
        }
        for k, sql in queries.items():
            r = self._tool(c, "sql_query", sql=sql, params=(lid, c.day))
            if not r.ok:
                return Obs(f"could not read {k}: {r.error}", ok=False, next=["abort"], replan="data unavailable; retry next cycle")
            out[k] = [list(row.values()) for row in r.data]
        c.activity = out
        return Obs(f"{len(out['logins'])} active days, {len(out['attendance'])} classes, {len(out['quizzes'])} quizzes")

    def _a_compute_risk(self, c: Ctx):
        args = {"day": c.day, "enrolled_day": c.learner["enrolled_day"], "logins": c.activity["logins"],
                "attendance": c.activity["attendance"], "quizzes": c.activity["quizzes"]}
        degraded = None
        if self.sandbox:
            r = self._tool(c, "analytics", fn="risk", args=args)
            risk = r.data if r.ok else None
            if not r.ok:
                degraded = r.error
        if not self.sandbox or degraded:
            risk = analytics.risk(**args)
        c.risk = risk
        if degraded:
            self._log(c.cycle, c.day, c.learner["learner_id"], "replan",
                      f"Sandbox failed ({degraded}); fell back to in-process analytics (degraded mode)")
        if risk["status"] == "insufficient_data":
            return Obs(f"only {risk['history_days']} days of history (< {config.MIN_HISTORY_DAYS})", next=["abstain"],
                       replan="not enough data to judge; abstain instead of guessing")
        flagged = risk["risk"] >= config.RISK_THRESHOLD
        msg = f"risk {risk['risk']:.2f} ({'≥' if flagged else '<'} {config.RISK_THRESHOLD})"
        if c.state.get("status") == "escalated":
            self.mem.save_state(c.learner["learner_id"], risk=risk["risk"], last_checked_day=c.day)
            return Obs(msg + "; case is with a human, monitoring only", data=risk, next=[])
        if not flagged:
            return Obs(msg, data=risk, next=["mark_monitoring"])
        return Obs(msg + " → at risk", data=risk,
                   next=["gather_evidence", "diagnose", "check_contact_policy", "choose_intervention",
                         "find_resources", "draft_message", "check_content", "submit_for_approval"],
                   replan="risk above threshold: investigate cause and prepare an intervention")

    def _a_abstain(self, c: Ctx):
        self.mem.save_state(c.learner["learner_id"], status="insufficient_data", risk=None, last_checked_day=c.day)
        return Obs("marked insufficient_data; will re-check when more history exists", next=[])

    def _a_mark_monitoring(self, c: Ctx):
        prev = c.state.get("status")
        status = "recovered" if prev == "recovered" else "monitoring"
        note = {}
        if c.diag and c.diag.get("explained_absence"):
            status, note = "watch", {"explanation": f"Explained absence: “{c.diag['absence_note']}”. Learner is active again."}
        self.mem.save_state(c.learner["learner_id"], status=status, risk=c.risk["risk"], last_checked_day=c.day, **note)
        return Obs(f"status → {status}", next=[])

    def _a_abort(self, c: Ctx):
        return Obs("episode aborted", ok=False, next=[])

    def _a_gather_evidence(self, c: Ctx):
        c.evidence = gather_evidence(_Bound(self, c), c.learner["learner_id"], c.day)
        f = c.evidence["failures"]
        return Obs(f"collected evidence from 6 sources" + (f"; {len(f)} failed" if f else ""), ok=not f, data={"failures": f})

    def _tried(self, lid):
        ivs = self.mem.interventions(lid)
        failed = [i["type"] for i in ivs if i["status"] == "measured" and not i["success"]]
        tried_all = [i["type"] for i in ivs if i["status"] in ("measured", "rejected", "sent")]
        return failed, tried_all

    def _a_diagnose(self, c: Ctx):
        lid = c.learner["learner_id"]
        failed, _ = self._tried(lid)
        d = diagnose(c.evidence, c.risk, failed, planner.PRIOR_EFFECT)
        c.diag = d
        probs = ", ".join(f"{CAUSE_LABEL[k]} {v:.0%}" for k, v in sorted(d["probs"].items(), key=lambda x: -x[1]))
        if d["explained_absence"]:
            return Obs(f"absence explained by learner (“{d['absence_note']}”) and activity has resumed", data=d,
                       next=["mark_monitoring"], replan="false alarm: planned absence, so no outreach. Keep watching")
        if d["unclear"]:
            c.escalation = "signals too weak to infer a cause"
            return Obs(f"no cause has enough evidence ({probs})", data=d, next=["escalate"],
                       replan="cause unclear; a human should talk to the learner rather than an automated guess")
        expl = drafting.explain(c.learner, c.risk, d)
        self.mem.save_state(lid, status="at_risk", risk=c.risk["risk"], cause=d["cause"], cause_probs=d["probs"],
                            explanation=expl, evidence=d["all_findings"], last_checked_day=c.day)
        extra = f" (after ruling out: {', '.join(failed)})" if failed else ""
        return Obs(f"probable cause: {CAUSE_LABEL[d['cause']]}{extra} | {probs}", data={"explanation": expl, **d})

    def _a_check_contact_policy(self, c: Ctx):
        rep = guardrails.check_contact(c.learner, self.mem.state(c.learner["learner_id"]), c.day)
        if not rep.allowed:
            c.escalation = "; ".join(v["detail"] for v in rep.violations)
            return Obs(f"🛑 refused automated outreach: {c.escalation}", ok=False, data=rep.as_dict(), phase="guardrail",
                       next=["escalate"], replan="outreach not permitted; escalate to a human counsellor instead")
        return Obs("contact allowed", data=rep.as_dict(), phase="guardrail")

    def _a_choose_intervention(self, c: Ctx):
        lid = c.learner["learner_id"]
        _, tried_all = self._tried(lid)
        st = self.mem.state(lid)
        res = planner.choose(c.diag["probs"], tried_all, self.mem.effectiveness(), st.get("attempts") or 0)
        if res["decision"] == "escalate":
            c.escalation = res["reason"]
            return Obs(f"no automated option: {res['reason']}", data=res, next=["escalate"], replan="escalate to a human")
        c.choice = res["choice"]
        alts = ", ".join(f"{o['type']} {o['p_success']:.0%}" for o in res["options"][1:3])
        return Obs(f"chose {c.choice['label']} (P(success)≈{c.choice['p_success']:.0%}; next best: {alts})", data=res)

    def _a_find_resources(self, c: Ctx):
        spec = planner.CATALOG[c.choice["type"]]
        topic = c.diag.get("stuck_topic") if c.choice["type"] in ("doubt_session", "easier_plan") else None
        q = spec["query"].format(topic=topic or "")
        r = self._tool(c, "search_resources", query=q, k=3, topic=topic)
        if r.ok and r.data:
            c.resources = r.data
            return Obs(f"found {len(r.data)} resources: " + "; ".join(x["title"] for x in r.data))
        r2 = self._tool(c, "search_resources", query=q, k=3)  # re-plan: broaden the search
        c.resources = r2.data if r2.ok and r2.data else []
        return Obs(f"topic-specific search empty/failed; broadened search → {len(c.resources)} results", ok=bool(c.resources))

    def _a_draft_message(self, c: Ctx):
        spec = planner.CATALOG[c.choice["type"]]
        session = {"day": c.day + 2} if spec["session"] else None
        c.draft = drafting.draft(c.choice["type"], c.learner, c.diag, c.resources, session)
        c.draft["session_day"] = session["day"] if session else None
        return Obs(f"drafted “{c.draft['subject']}” ({c.draft['drafted_by']})")

    def _a_check_content(self, c: Ctx):
        rep = guardrails.check_content(c.draft["body"] + "\n" + c.draft["subject"], c.learner, self._learners())
        c.guard = rep.as_dict()
        if rep.allowed:
            return Obs("content policy passed" + (f" ({len(rep.warnings)} warnings)" if rep.warnings else ""),
                       data=c.guard, phase="guardrail")
        if c.draft["drafted_by"] == "nemotron":
            subj, body = drafting.template(c.choice["type"], c.learner, c.diag, c.resources,
                                           {"day": c.draft["session_day"]} if c.draft["session_day"] else None)
            c.draft.update(subject=subj, body=body, drafted_by="template (LLM draft failed policy)")
            return Obs(f"🛑 LLM draft violated policy: {rep.violations}", ok=False, data=c.guard, phase="guardrail",
                       next=["check_content", "submit_for_approval"], replan="discard LLM draft, use the vetted template")
        c.escalation = "draft failed content policy"
        return Obs(f"🛑 draft violated policy: {rep.violations}", ok=False, data=c.guard, phase="guardrail", next=["escalate"])

    def _a_submit_for_approval(self, c: Ctx):
        lid = c.learner["learner_id"]
        iid = self.mem.add_intervention(
            learner_id=lid, cause=c.diag["cause"], type=c.choice["type"], status="pending_approval",
            rationale=f"{c.choice['label']} for {CAUSE_LABEL[c.diag['cause']]} (P(success)≈{c.choice['p_success']:.0%})",
            subject=c.draft["subject"], body=c.draft["body"], resources=c.resources, created_day=c.day,
            guardrail=c.guard, drafted_by=c.draft["drafted_by"], session_day=c.draft["session_day"],
            explanation=self.mem.state(lid).get("explanation"),
            baseline={"risk": c.risk["risk"], "cause_probs": c.diag["probs"]})
        self.mem.save_state(lid, status="awaiting_approval")
        return Obs(f"⏸ intervention #{iid} queued for human approval", data={"intervention_id": iid}, phase="approval", next=[])

    def _a_escalate(self, c: Ctx):
        lid = c.learner["learner_id"]
        self.mem.save_state(lid, status="escalated", risk=(c.risk or {}).get("risk"), last_checked_day=c.day,
                            cause=(c.diag or {}).get("cause"), cause_probs=(c.diag or {}).get("probs"))
        self.mem.add_intervention(learner_id=lid, cause=(c.diag or {}).get("cause"), type="human_escalation",
                                  status="escalated", rationale=c.escalation, created_day=c.day,
                                  subject="Needs human follow-up", body=c.escalation,
                                  explanation=drafting.explain(c.learner, c.risk, c.diag) if c.diag else c.escalation)
        return Obs(f"handed to human counsellor queue: {c.escalation}", phase="approval", next=[])

    def _a_measure_outcome(self, c: Ctx):
        lid = c.learner["learner_id"]
        iv = [i for i in self.mem.interventions(lid, status="sent")][-1]
        s = iv["sent_day"]
        base = dict(enrolled_day=c.learner["enrolled_day"], logins=c.activity["logins"],
                    attendance=c.activity["attendance"], quizzes=c.activity["quizzes"])
        windows = {"healthy_baseline": (max(c.learner["enrolled_day"], s - 28), s - 14), "before": (s - 7, s), "after": (s + 3, c.day)}
        ei = {}
        for name, (a, b) in windows.items():
            r = self._tool(c, "analytics", fn="engagement", args={"a": a, "b": b, **base}) if self.sandbox else None
            ei[name] = r.data if (r and r.ok) else analytics.engagement(a, b, **base)
        pre, post, healthy = ei["before"]["index"], ei["after"]["index"], ei["healthy_baseline"]["index"]
        gap = max(healthy - pre, .05)
        ratio = round((post - pre) / gap, 2)
        success = ratio >= .5 or post >= .9 * healthy
        outcome = {"engagement": ei, "recovery_ratio": ratio, "delta": round(post - pre, 3)}
        self.mem.update_intervention(iv["id"], status="measured", outcome=outcome, success=int(success))
        self.mem.record_outcome(iv["cause"], iv["type"], success)
        label = planner.CATALOG[iv["type"]]["label"]
        summary = (f"{label}: engagement {pre:.2f} → {post:.2f} (healthy {healthy:.2f}), "
                   f"closed {'100%+' if ratio > 1 else f'{ratio:.0%}'} of the gap → {'SUCCESS' if success else 'NOT WORKING'}")
        if success:
            self.mem.save_state(lid, status="recovered")
            return Obs(summary, data=outcome, phase="outcome", next=[])
        self.mem.save_state(lid, status="at_risk")
        c.state = self.mem.state(lid)
        return Obs(summary, ok=False, data=outcome, phase="outcome",
                   next=["compute_risk_for_replan", "gather_evidence", "diagnose", "check_contact_policy",
                         "choose_intervention", "find_resources", "draft_message", "check_content", "submit_for_approval"],
                   replan=f"'{label}' did not work. Down-weight the causes it should have fixed "
                          f"(Bayesian update), re-diagnose with fresh data and pick a different intervention")

    def _a_compute_risk_for_replan(self, c: Ctx):
        args = {"day": c.day, "enrolled_day": c.learner["enrolled_day"], **c.activity}
        r = self._tool(c, "analytics", fn="risk", args=args) if self.sandbox else None
        c.risk = r.data if (r and r.ok) else analytics.risk(**args)
        return Obs(f"current risk {c.risk['risk']:.2f}")

    # ================================================================== execution (post-approval)
    def _execute(self, iid, day, cycle):
        iv = self.mem.intervention(iid)
        guardrails.require_approval(iv)   # hard gate: nothing is sent without a named approver
        learner = self._learner(iv["learner_id"])
        spec = planner.CATALOG[iv["type"]]
        lid = learner["learner_id"]
        session_file = iv.get("session_file")
        if spec["session"] and not session_file:
            r = self.tools.call("schedule_session", learner=learner, kind=spec["session"][0], host_role=spec["session"][1],
                                day=iv["session_day"] or day + 2)
            self._log(cycle, day, lid, "act", f"schedule_session → {'ok' if r.ok else r.error}", r.data)
            if r.ok:
                session_file = r.data["file"]
                self.mem.update_intervention(iid, session_file=session_file)
        r = self.tools.call("send_message", to=learner["email"], subject=iv["subject"], body=iv["body"],
                            attachments=[session_file], notify=f"[EDA] {spec['label']} sent to {lid} (approved by {iv['reviewer']})")
        how = "REAL email delivered" if r.ok and (r.data or {}).get("real") else "delivered"
        self._log(cycle, day, lid, "act", f"send_message via SMTP → {how if r.ok else 'FAILED'} "
                  f"(attempts {r.attempts}, {r.latency_ms} ms)", {"error": r.error, **(r.data or {})})
        if not r.ok:
            self.mem.update_intervention(iid, status="send_failed")
            self._log(cycle, day, lid, "replan", "Delivery failed after retries. Kept in queue, will retry next cycle; "
                      "learner state unchanged (no follow-up clock started)", {"error": r.error})
            return {"ok": False, "status": "send_failed", "error": r.error}
        st = self.mem.state(lid)
        self.mem.update_intervention(iid, status="sent", sent_day=day, followup_day=day + config.FOLLOWUP_DAYS)
        self.mem.save_state(lid, status="intervention_active", last_contact_day=day, attempts=(st.get("attempts") or 0) + 1)
        return {"ok": True, "status": "sent", "followup_day": day + config.FOLLOWUP_DAYS, "session_file": session_file}

    # ================================================================== helpers
    def _tool(self, c: Ctx, name, **kw):
        r = self.tools.call(name, **kw)
        brief = kw.get("sql", kw.get("query", kw.get("fn", "")))
        brief = " ".join(str(brief).split())[:90]
        self._log(c.cycle, c.day, c.learner["learner_id"], "act",
                  f"{name}({brief}) → {'ok' if r.ok else 'ERROR: ' + str(r.error)} [{r.latency_ms} ms]")
        return r

    def _log(self, cycle, day, lid, phase, title, detail=None):
        self.mem.log(cycle, day, lid, phase, title, detail)
        if self.verbose:
            print(f"  [{phase:9}] {lid or '-':5} {title}")

    def _learners(self):
        if self._cohort is None:
            r = self.tools.call("sql_query", sql="SELECT * FROM learners ORDER BY learner_id")
            if not r.ok:
                raise RuntimeError(r.error)
            self._cohort = [{**x, "first_name": x["name"].split()[0]} for x in r.data]
        return self._cohort

    def _learner(self, lid):
        return next(x for x in self._learners() if x["learner_id"] == lid)


class _Bound:
    """Adapter so diagnoser tool calls are logged against the current learner."""

    def __init__(self, agent, ctx):
        self.agent, self.ctx = agent, ctx

    def call(self, name, **kw):
        return self.agent._tool(self.ctx, name, **kw)