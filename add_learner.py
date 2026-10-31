"""Add a learner to the current simulated world.

The agent never stores a risk score you can set: it recomputes risk each cycle from activity.
So a learner is added with a simulated history (archetype, onset day, seed), and the agent
scores them on the next cycle like everyone else.

Example (stuck on Functions, risk 0.78 at day 28):
    python add_learner.py --id L041 --name "Mahesh Channa" --archetype stuck --onset 15 --seed 296

To really email a learner, give their address with --email, list it in EDA_REAL_RECIPIENTS and
configure EDA_RELAY_* in .env (see README). Unlisted addresses only ever reach the local sink.
"""
from __future__ import annotations

import argparse

from eda import analytics
from eda.world import ARCH_PARAMS, LearnerSpec, World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--email", help="default: first.last@learners.example (simulated, never really emailed)")
    ap.add_argument("--archetype", default="stuck", choices=["healthy", *ARCH_PARAMS])
    ap.add_argument("--onset", type=int, default=15, help="day disengagement starts")
    ap.add_argument("--seed", type=int, default=0, help="seed for this learner's simulated behaviour")
    ap.add_argument("--plan", default="upfront", choices=["upfront", "emi"])
    args = ap.parse_args()

    first, last = args.name.split()[0], args.name.split()[-1]
    email = args.email or f"{first.lower()}.{last.lower()}@learners.example"
    spec = LearnerSpec(args.id, args.name, email, args.archetype,
                       plan=args.plan, onset=args.onset, seed=args.seed, ability=79.0, base_login=.86,
                       do_not_contact=int(args.archetype == "dnc"))
    world = World()
    o = world.add_learner(spec)
    r = analytics.risk(world.clock, spec.enrolled_day, [x[1:3] for x in o.logins], [x[1:] for x in o.attendance],
                       [x[1:] for x in o.quizzes])
    print(f"Added {args.id} {args.name} on day {world.clock}: risk {r['risk']} "
          f"(archetype {args.archetype}{', stuck on ' + o.stuck_topic if o.stuck_topic else ''}). "
          f"Press ▶ Run cycle and the agent will pick them up.")


if __name__ == "__main__":
    main()
