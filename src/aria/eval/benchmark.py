"""A synthetic multi-turn memory-recall benchmark.

Each scenario plants "needle" facts early, buries them under distractor turns, then
probes recall at the end. Conversation length (and thus the planted-fact → probe
distance) grows across scenarios, which is exactly what separates a bounded
sliding-window from true long-term memory. Generation is seeded and deterministic so
results are reproducible in CI.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# (attribute, recall question, expected needle value)
_FACTS: list[tuple[str, str, str]] = [
    ("favorite color", "What is my favorite color?", "teal"),
    ("name", "What is my name?", "Sam"),
    ("hometown", "Where am I from?", "Lisbon"),
    ("pet", "What is my pet's name?", "Pixel"),
    ("job", "What do I do for work?", "data scientist"),
    ("project", "What am I building?", "recommender system"),
    ("allergy", "What am I allergic to?", "peanuts"),
    ("language", "What is my favorite programming language?", "Python"),
]
_STATEMENTS: dict[str, str] = {
    "favorite color": "My favorite color is teal.",
    "name": "My name is Sam.",
    "hometown": "I'm from Lisbon.",
    "pet": "My pet is a corgi named Pixel.",
    "job": "I work as a data scientist.",
    "project": "I'm building a recommender system.",
    "allergy": "I'm allergic to peanuts.",
    "language": "My favorite programming language is Python.",
}
_FILLERS: list[str] = [
    "The weather has been unpredictable lately.",
    "I watched an interesting documentary yesterday.",
    "Let's chat about something else for a moment.",
    "I went for a long walk this morning.",
    "The traffic was terrible on the way here.",
    "I tried a new recipe over the weekend.",
    "There is a concert in town next week.",
    "My commute felt longer than usual today.",
    "I reorganised my bookshelf last night.",
    "The coffee shop downstairs changed its menu.",
]
_ACK = "Got it, thanks for telling me."
_PROBE_REPLY = "Let me recall that for you."


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    text: str
    kind: str = "filler"  # "fact" | "filler" | "probe" | "ack"
    needle: str | None = None  # expected substring, for probe turns


@dataclass
class Scenario:
    id: str
    turns: list[Turn] = field(default_factory=list)

    @property
    def probes(self) -> list[Turn]:
        return [t for t in self.turns if t.kind == "probe"]


def generate_benchmark(
    n_scenarios: int = 8, facts_per_scenario: int = 3, seed: int = 0
) -> list[Scenario]:
    """Generate a deterministic list of recall scenarios with growing distances."""
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for i in range(n_scenarios):
        chosen = rng.sample(_FACTS, k=min(facts_per_scenario, len(_FACTS)))
        n_fillers = 2 + i * 2  # distance grows with scenario index
        turns: list[Turn] = []

        for attribute, _question, value in chosen:
            turns.append(Turn("user", _STATEMENTS[attribute], "fact", value))
            turns.append(Turn("assistant", _ACK, "ack"))

        for _ in range(n_fillers):
            turns.append(Turn("user", rng.choice(_FILLERS), "filler"))
            turns.append(Turn("assistant", _ACK, "ack"))

        for _attribute, question, value in chosen:
            turns.append(Turn("user", question, "probe", value))
            turns.append(Turn("assistant", _PROBE_REPLY, "ack"))

        scenarios.append(Scenario(id=f"scenario-{i}", turns=turns))
    return scenarios


__all__ = ["Scenario", "Turn", "generate_benchmark"]
