"""Streamlit dashboard for Rekindle, the Early-Disengagement Detection and Re-engagement Agent.

    streamlit run app.py
"""
from __future__ import annotations

import email
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from eda import config, drafting
from eda.diagnoser import CAUSE_LABEL
from eda.planner import CATALOG
from eda.runner import Runner
from eda.tools import ToolBox, day_to_date, sql_query

st.set_page_config(page_title="Rekindle", page_icon="🎓", layout="wide")
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"
STATUS_ICON = {"monitoring": "🟢", "recovered": "💚", "watch": "🟡", "insufficient_data": "⚪", "at_risk": "🔴",
               "awaiting_approval": "⏸️", "intervention_active": "🔵", "escalated": "🧑‍💼"}
PHASE_ICON = {"plan": "🗺️", "act": "🔧", "observe": "👀", "replan": "🔁", "guardrail": "🛡️", "approval": "✅", "outcome": "📈"}


@st.cache_resource
def get_runner():
    r = Runner()
    try:
        sql_query("SELECT 1 FROM learners LIMIT 1")
    except Exception:
        r.reset(7)
    return r


run = get_runner()
agent, mem = run.agent, run.mem

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎓 Rekindle")
    st.metric("Simulation day", run.day, help=f"Day 0 = {day_to_date(0):%d %b %Y}")
    st.caption(f"Today: {day_to_date(run.day):%A %d %b %Y}")
    reviewer = st.text_input("Reviewer (you)", "Coach Asha")
    c1, c2 = st.columns(2)
    if c1.button("▶ Run cycle", width="stretch", type="primary"):
        with st.spinner("Agent planning → acting → observing ..."):
            run.run_cycle()
        st.rerun()
    if c2.button(f"⏩ +{config.FOLLOWUP_DAYS} days", width="stretch"):
        with st.spinner("Advancing the world and running a cycle ..."):
            run.advance(config.FOLLOWUP_DAYS)
            run.run_cycle()
        st.rerun()
    outage = st.toggle("Simulate mail-server outage", value=st.session_state.get("outage", False),
                       help="Points send_message at a dead SMTP port to demonstrate retries and failure handling")
    if outage != st.session_state.get("outage", False):
        st.session_state.outage = outage
        agent.tools = ToolBox(smtp_port=1, retry_backoff=(0.2, 0.4)) if outage else ToolBox()
    st.divider()
    seed = st.number_input("Cohort seed", value=7, step=1)
    if st.button("↺ Reset world & memory", width="stretch"):
        run.reset(int(seed))
        st.rerun()
    st.divider()
    st.caption(f"**Drafting:** {drafting.llm_status()}")
    st.caption(f"**SMTP:** {config.SMTP_HOST}:{config.SMTP_PORT}" + (" (local sink)" if run.mailsink else ""))
    st.caption("**Tools:** sql_query · run_python sandbox · search_resources · send_message (SMTP) · schedule_session (.ics)")

learners = pd.DataFrame(sql_query("SELECT learner_id, name, email, plan, do_not_contact FROM learners"))
states = {s["learner_id"]: s for s in mem.all_states()}
ivs = mem.interventions()
pending = [i for i in ivs if i["status"] == "pending_approval"]

tabs = st.tabs([f"📡 Risk radar", f"✅ Approval queue ({len(pending)})", "🔍 Learner", "📈 Outcomes", "🧠 Agent trace",
                "🧪 Evaluation", "✉️ Outbox"])

# ------------------------------------------------------------------ risk radar
with tabs[0]:
    if not states:
        st.info("No cycle run yet. Press **▶ Run cycle** in the sidebar.")
    else:
        counts = pd.Series([s["status"] for s in states.values()]).value_counts()
        cols = st.columns(6)
        for col, (label, keys) in zip(cols, [("Monitored", None), ("Needs approval", ["awaiting_approval"]),
                                              ("Intervention active", ["intervention_active"]), ("Recovered", ["recovered"]),
                                              ("With a human", ["escalated"]), ("Watch / no data", ["watch", "insufficient_data"])]):
            col.metric(label, len(states) if keys is None else int(sum(counts.get(k, 0) for k in keys)))
        rows = []
        for _, lr in learners.iterrows():
            s = states.get(lr.learner_id, {})
            probs = s.get("cause_probs") or {}
            rows.append({"": STATUS_ICON.get(s.get("status"), ""), "Learner": lr.learner_id, "Name": lr["name"],
                         "Status": s.get("status", "-"), "Risk": s.get("risk"),
                         "Probable cause": CAUSE_LABEL.get(s.get("cause"), "") if s.get("status") not in ("monitoring", "watch") else "",
                         "Confidence": max(probs.values()) if probs and s.get("status") not in ("monitoring", "watch") else None,
                         "Checked on day": s.get("last_checked_day")})
        df = pd.DataFrame(rows).sort_values("Risk", ascending=False, na_position="last")
        st.dataframe(df, hide_index=True, width="stretch", height=620, column_config={
            "Risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        st.caption(f"Flag threshold: risk ≥ {config.RISK_THRESHOLD}. Risk combines drops vs the learner's own baseline in "
                   "logins, attendance and quiz scores, inactivity, weekly trend, repeated failures, and absolute activity.")

# ------------------------------------------------------------------ approval queue
with tabs[1]:
    st.subheader("Human approval checkpoint")
    st.caption("Nothing is sent until a named reviewer approves. Edits are re-checked against the content policy before sending.")
    if not pending:
        st.success("Queue is empty.")
    for iv in pending:
        name = learners.set_index("learner_id").loc[iv["learner_id"], "name"]
        with st.expander(f"#{iv['id']} · {iv['learner_id']} {name} · {CATALOG[iv['type']]['label']}", expanded=iv is pending[0]):
            left, right = st.columns([3, 2])
            with left:
                subject = st.text_input("Subject", iv["subject"], key=f"s{iv['id']}")
                body = st.text_area("Message (editable)", iv["body"], height=330, key=f"b{iv['id']}")
                note = st.text_input("Reviewer note (optional)", key=f"n{iv['id']}")
                b1, b2 = st.columns(2)
                if b1.button("✅ Approve & send", key=f"a{iv['id']}", type="primary", width="stretch"):
                    res = agent.review(iv["id"], "approve", reviewer, body=body, subject=subject, note=note)
                    if res.get("refused"):
                        st.session_state[f"err{iv['id']}"] = res["violations"]
                    elif not res.get("ok"):
                        st.session_state[f"err{iv['id']}"] = [{"rule": "delivery", "detail": f"send failed: {res.get('error')}; will retry next cycle"}]
                    st.rerun()
                if b2.button("❌ Reject", key=f"r{iv['id']}", width="stretch"):
                    agent.review(iv["id"], "reject", reviewer, note=note)
                    st.rerun()
                for v in st.session_state.get(f"err{iv['id']}", []):
                    st.error(f"🛑 **Refused** ({v['rule']}): {v['detail']}")
            with right:
                st.markdown(f"**Why:** {iv['rationale']}")
                st.info(iv["explanation"] or "")
                probs = (iv["baseline"] or {}).get("cause_probs", {})
                if probs:
                    pdf = pd.DataFrame({"cause": [CAUSE_LABEL[c] for c in probs], "p": list(probs.values())})
                    st.altair_chart(alt.Chart(pdf).mark_bar(cornerRadiusEnd=4, height=14, color=BLUE).encode(
                        x=alt.X("p:Q", axis=alt.Axis(format="%", title="probability", grid=False), scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("cause:N", sort="-x", title=None), tooltip=["cause", alt.Tooltip("p:Q", format=".0%")]
                    ).properties(height=130), width="stretch")
                g = iv["guardrail"] or {}
                st.markdown("**Content policy:** " + ("✅ passed" if g.get("allowed") else "🛑 failed") +
                            "".join(f"\n- ⚠️ {w['detail']}" for w in g.get("warnings", [])))
                st.caption(f"Drafted by: {iv['drafted_by']}" + (f" · session on day {iv['session_day']} "
                           f"({day_to_date(iv['session_day']):%a %d %b})" if iv["session_day"] else ""))
    esc = [i for i in ivs if i["status"] == "escalated"]
    failed = [i for i in ivs if i["status"] == "send_failed"]
    if failed:
        st.warning(f"📮 {len(failed)} approved message(s) failed delivery and will be retried next cycle: "
                   + ", ".join(f"#{i['id']} {i['learner_id']}" for i in failed))
    if esc:
        st.subheader("🧑‍💼 Human counsellor queue (no automated outreach)")
        st.dataframe(pd.DataFrame([{"Learner": i["learner_id"], "Day": i["created_day"], "Reason": i["rationale"],
                                    "Case note": i["explanation"]} for i in esc]), hide_index=True, width="stretch")

# ------------------------------------------------------------------ learner deep dive
with tabs[2]:
    order = sorted(learners.learner_id, key=lambda l: -(states.get(l, {}).get("risk") or 0))
    lid = st.selectbox("Learner", order, format_func=lambda l: f"{l} · {learners.set_index('learner_id').loc[l, 'name']} "
                       f"· {states.get(l, {}).get('status', '-')}")
    s = states.get(lid, {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk", f"{s['risk']:.2f}" if s.get("risk") is not None else "-")
    c2.metric("Status", s.get("status", "-"))
    c3.metric("Interventions tried", s.get("attempts") or 0)
    if s.get("explanation"):
        st.info(s["explanation"])
    lg = pd.DataFrame(sql_query("SELECT day, minutes FROM logins WHERE learner_id=?", (lid,)))
    at = pd.DataFrame(sql_query("SELECT day, attended FROM attendance WHERE learner_id=?", (lid,)))
    qz = pd.DataFrame(sql_query("SELECT day, topic, score, attempt FROM quiz_scores WHERE learner_id=?", (lid,)))
    weeks = pd.DataFrame({"week_start": range(0, run.day, 7)})
    weeks["Login rate"] = [((lg.day >= w) & (lg.day < min(w + 7, run.day))).sum() / max(1, min(7, run.day - w)) if len(lg) else 0 for w in weeks.week_start]
    weeks["Attendance"] = [at[(at.day >= w) & (at.day < w + 7)].attended.mean() if len(at) else None for w in weeks.week_start]
    long = weeks.melt("week_start", var_name="series", value_name="rate").dropna()
    sent = pd.DataFrame([{"day": i["sent_day"], "what": CATALOG[i["type"]]["label"]} for i in ivs
                         if i["learner_id"] == lid and i["sent_day"] is not None])
    left, right = st.columns(2)
    with left:
        base = alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
            x=alt.X("week_start:Q", title="day (week start)"), y=alt.Y("rate:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
            color=alt.Color("series:N", scale=alt.Scale(range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["series", "week_start", alt.Tooltip("rate:Q", format=".0%")])
        layers = [base]
        if len(sent):
            layers.append(alt.Chart(sent).mark_rule(strokeDash=[4, 3], color=MUTED).encode(x="day:Q", tooltip=["what", "day"]))
        st.markdown("**Weekly engagement** (dashed = intervention sent)")
        st.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")
    with right:
        st.markdown("**Quiz scores** (line = 55% fail threshold)")
        if len(qz):
            pts = alt.Chart(qz).mark_circle(size=70, color=BLUE, opacity=.85).encode(
                x=alt.X("day:Q"), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title=None),
                tooltip=["day", "topic", "score", "attempt"])
            rule = alt.Chart(pd.DataFrame({"y": [55]})).mark_rule(color=MUTED, strokeDash=[2, 2]).encode(y="y:Q")
            layers = [pts, rule] + ([alt.Chart(sent).mark_rule(strokeDash=[4, 3], color=MUTED).encode(x="day:Q")] if len(sent) else [])
            st.altair_chart(alt.layer(*layers).properties(height=260), width="stretch")
    if s.get("evidence"):
        st.markdown("**Evidence by cause**")
        ecols = st.columns(4)
        for col, c in zip(ecols, CAUSE_LABEL):
            p = (s.get("cause_probs") or {}).get(c, 0)
            col.markdown(f"**{CAUSE_LABEL[c]}** · {p:.0%}")
            for f in s["evidence"].get(c, []):
                col.caption(f"• {f}")
    hist = [i for i in ivs if i["learner_id"] == lid]
    if hist:
        st.markdown("**Intervention history**")
        st.dataframe(pd.DataFrame([{"#": i["id"], "type": i["type"], "status": i["status"], "created": i["created_day"],
                                    "sent": i["sent_day"], "reviewer": i["reviewer"],
                                    "recovery": (i["outcome"] or {}).get("recovery_ratio"),
                                    "success": None if i["success"] is None else bool(i["success"])} for i in hist]),
                     hide_index=True, width="stretch")
    with st.expander("Agent trace for this learner"):
        for e in mem.trace(lid):
            st.markdown(f"`day {e['day']}` {PHASE_ICON.get(e['phase'], '')} **{e['phase']}**: {e['title']}")

# ------------------------------------------------------------------ outcomes
with tabs[3]:
    measured = [i for i in ivs if i["status"] == "measured"]
    if not measured:
        st.info(f"No outcomes yet. Approve interventions, then advance {config.FOLLOWUP_DAYS} days.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Interventions measured", len(measured))
        c2.metric("Worked", f"{sum(i['success'] for i in measured) / len(measured):.0%}")
        real = [i for i in ivs if i["type"] != "human_escalation"]
        c3.metric("Re-plans after failure", len(real) - len({i["learner_id"] for i in real}))
        rows = [{"Learner": i["learner_id"], "Intervention": CATALOG[i["type"]]["label"], "Diagnosed cause": CAUSE_LABEL.get(i["cause"]),
                 "Sent day": i["sent_day"], "Before": i["outcome"]["engagement"]["before"]["index"],
                 "After": i["outcome"]["engagement"]["after"]["index"], "Healthy baseline": i["outcome"]["engagement"]["healthy_baseline"]["index"],
                 "Gap closed": i["outcome"]["recovery_ratio"], "Result": "✅ recovered" if i["success"] else "❌ re-plan"} for i in measured]
        odf = pd.DataFrame(rows)
        st.dataframe(odf, hide_index=True, width="stretch")
        dumb = odf.melt(["Learner", "Result"], ["Before", "After"], var_name="when", value_name="engagement")
        line = alt.Chart(dumb).mark_line(color=MUTED, strokeWidth=2).encode(x=alt.X("engagement:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("Learner:N", title=None), detail="Learner:N")
        pts = alt.Chart(dumb).mark_circle(size=90, opacity=1).encode(
            x="engagement:Q", y="Learner:N", color=alt.Color("when:N", scale=alt.Scale(domain=["Before", "After"], range=[ORANGE, BLUE]),
                                                               legend=alt.Legend(orient="top", title=None)),
            tooltip=["Learner", "when", alt.Tooltip("engagement:Q", format=".2f"), "Result"])
        st.markdown("**Engagement index before vs after intervention**")
        st.altair_chart((line + pts).properties(height=max(200, 22 * len(odf))), width="stretch")
        eff = mem.effectiveness()
        if eff:
            st.markdown("**Learned effectiveness (memory, persists across runs)**")
            st.dataframe(pd.DataFrame([{"cause": CAUSE_LABEL[c], "intervention": t, "successes": s_, "trials": n,
                                        "rate": s_ / n} for (c, t), (s_, n) in sorted(eff.items())]), hide_index=True)

# ------------------------------------------------------------------ trace
with tabs[4]:
    c1, c2 = st.columns(2)
    phases = c1.multiselect("Phases", list(PHASE_ICON), default=["plan", "observe", "replan", "guardrail", "approval", "outcome"])
    who = c2.selectbox("Learner filter", ["(all)"] + list(learners.learner_id))
    tr = [e for e in mem.trace(None if who == "(all)" else who, limit=3000) if e["phase"] in phases]
    st.caption(f"{len(tr)} steps (most recent last)")
    st.dataframe(pd.DataFrame([{"cycle": e["cycle"], "day": e["day"], "learner": e["learner_id"],
                                "phase": f"{PHASE_ICON.get(e['phase'], '')} {e['phase']}", "step": e["title"]} for e in tr[-600:]]),
                 hide_index=True, width="stretch", height=640)

# ------------------------------------------------------------------ evaluation
with tabs[5]:
    st.caption("Runs `evaluate.py` in an isolated data directory (does not touch this demo's state).")
    if st.button("Run evaluation (~35 s)"):
        with st.spinner("Evaluating ..."):
            p = subprocess.run([sys.executable, "evaluate.py"], capture_output=True, text=True, cwd=config.ROOT)
        if p.returncode:
            st.error(p.stderr[-2000:])
    rep = config.ROOT / "EVAL_REPORT.md"
    if rep.exists():
        st.markdown(rep.read_text())

# ------------------------------------------------------------------ outbox
with tabs[6]:
    files = sorted(Path(config.OUTBOX_DIR).glob("*.eml"), reverse=True)
    st.caption(f"{len(files)} emails received by the SMTP sink · invites in {config.CALENDAR_DIR}")
    for f in files[:50]:
        m = email.message_from_bytes(f.read_bytes())
        with st.expander(f"{m['To']} · {m['Subject']}"):
            st.text(m.get_payload(decode=True).decode(errors="replace") if not m.is_multipart() else m.get_payload()[0].get_payload())
