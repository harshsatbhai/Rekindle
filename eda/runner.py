"""Wires the world (environment), the agent and the mail sink together."""
from __future__ import annotations

import shutil

from . import config, mailsink
from .agent import Agent
from .memory import Memory
from .world import World


class Runner:
    def __init__(self, verbose=False, smtp_port=None, start_mailsink=True):
        self.world = World()
        self.mem = Memory()
        self.agent = Agent(self.mem, smtp_port=smtp_port, verbose=verbose)
        self.mailsink = mailsink.start() if start_mailsink else None

    def reset(self, seed=7, day=config.DETECTION_DAY):
        for d in (config.OUTBOX_DIR, config.CALENDAR_DIR):
            shutil.rmtree(d, ignore_errors=True)
        self.world.reset(seed, day)
        self.mem.reset()
        self.agent._cohort = None
        self.mem.set_meta("day", day)

    @property
    def day(self):
        return self.world.clock

    def run_cycle(self):
        # make sure the world reflects every delivered intervention before the agent looks
        self.world.advance_to(self.day, self.mem.delivered())
        return self.agent.run_cycle(self.day)

    def advance(self, days: int):
        self.world.advance_to(self.day + days, self.mem.delivered())
        self.mem.set_meta("day", self.day)
        return self.day
