"""
A light-weight synchronous FSM using the `transitions` library.
It mirrors the LangGraph for cases where you do NOT want an LLM
but still need deterministic workflow orchestration.
"""

from transitions import Machine


class ComplaintFSM:
    states = ["collect", "classify", "confirm", "execute", "done"]

    def __init__(self):
        self.data = {}
        Machine(model=self, states=self.states, initial="collect")

        self.add_transition("collected", "collect", "classify")
        self.add_transition("classified", "classify", "confirm")
        self.add_transition("accepted", "confirm", "execute")
        self.add_transition("rejected", "confirm", "collect")
        self.add_transition("executed", "execute", "done")

    def on_enter_collect(self): ...
    def on_enter_classify(self): ...
    def on_enter_confirm(self): ...
    def on_enter_execute(self): ...
