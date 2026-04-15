from __future__ import annotations

from .Agent import Agent


class TeamAgent(Agent):
    def _build_participant_instructions(self, extra_instructions: str | None) -> str:
        return self._build_instructions(extra_instructions)
