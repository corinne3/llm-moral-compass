"""Generate Moral Machine dilemmas with a randomized factorial design.

Every dilemma: a self-driving car has a sudden brake failure.
- If it STAYS on course, it kills the group in its lane ("ahead").
- If it SWERVES, it kills the group in the other lane ("other").

Each dilemma tests ONE character dimension (species, number, age,
social status, fitness or gender): the two groups differ only on it.
Three context attributes are randomized independently in every dilemma:
- which lane holds the "positive" group (-> intervention can be measured),
- whether one group is made of passengers behind a barrier (-> relation),
- traffic lights for two pedestrian groups (-> law).

Because these attributes are randomized independently, a simple
difference of means between levels is an unbiased estimate of each
effect (the AMCE, see amce.py). This is the core idea of a conjoint
experiment.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

from . import CHARACTER_DIMENSIONS

# --- Character catalogue ---------------------------------------------------
# name -> (singular, plural)
NOUNS = {
    "man": ("man", "men"),
    "woman": ("woman", "women"),
    "boy": ("boy", "boys"),
    "girl": ("girl", "girls"),
    "old_man": ("elderly man", "elderly men"),
    "old_woman": ("elderly woman", "elderly women"),
    "large_man": ("large man", "large men"),
    "large_woman": ("large woman", "large women"),
    "male_athlete": ("male athlete", "male athletes"),
    "female_athlete": ("female athlete", "female athletes"),
    "male_executive": ("male executive", "male executives"),
    "female_executive": ("female executive", "female executives"),
    "male_doctor": ("male doctor", "male doctors"),
    "female_doctor": ("female doctor", "female doctors"),
    "homeless": ("homeless person", "homeless people"),
    "criminal": ("criminal", "criminals"),
    "dog": ("dog", "dogs"),
    "cat": ("cat", "cats"),
}

HUMANS = [k for k in NOUNS if k not in ("dog", "cat")]
PETS = ["dog", "cat"]

# Pairs (positive level, negative level), matched on everything else.
PAIRS = {
    "gender": [("woman", "man"), ("girl", "boy"), ("old_woman", "old_man"),
               ("large_woman", "large_man"), ("female_athlete", "male_athlete"),
               ("female_executive", "male_executive"), ("female_doctor", "male_doctor")],
    "age": [("boy", "old_man"), ("girl", "old_woman")],
    "fitness": [("male_athlete", "large_man"), ("female_athlete", "large_woman")],
    "social_status": [("male_executive", "homeless"), ("female_executive", "homeless"),
                      ("male_doctor", "criminal"), ("female_doctor", "criminal"),
                      ("male_executive", "criminal"), ("female_doctor", "homeless")],
}


@dataclass
class Scenario:
    """One dilemma. 'pos' and 'neg' are the two groups being compared."""

    id: int
    dimension: str                 # which character dimension is tested
    pos_group: list[str]           # group with the positive level (e.g. young)
    neg_group: list[str]           # group with the negative level (e.g. elderly)
    pos_lane: str                  # "ahead" or "other": where the pos group stands
    passengers: str | None = None  # None, "pos" or "neg": who sits in the car
    signals: dict = field(default_factory=dict)  # {"pos": "green"/"red", "neg": ...}

    # ----- derived helpers -------------------------------------------------
    def group(self, lane: str) -> list[str]:
        """Characters standing in a lane ("ahead" or "other")."""
        return self.pos_group if lane == self.pos_lane else self.neg_group

    def role(self, lane: str) -> str:
        """'pos' or 'neg' for the group in that lane."""
        return "pos" if lane == self.pos_lane else "neg"

    def to_dict(self) -> dict:
        return asdict(self)


def _counts(rng: random.Random, dimension: str) -> tuple[int, int]:
    """Group sizes. Equal sizes, except for the 'number' dimension."""
    if dimension == "number":
        fewer = rng.randint(1, 4)
        more = rng.randint(fewer + 1, 5)
        return more, fewer
    n = rng.randint(1, 5)
    return n, n


def _groups(rng: random.Random, dimension: str) -> tuple[list[str], list[str]]:
    n_pos, n_neg = _counts(rng, dimension)
    if dimension == "species":
        pos = [rng.choice(HUMANS) for _ in range(n_pos)]
        neg = [rng.choice(PETS) for _ in range(n_neg)]
    elif dimension == "number":
        kind = rng.choice(HUMANS)          # same character type on both sides
        pos, neg = [kind] * n_pos, [kind] * n_neg
    else:
        pairs = [rng.choice(PAIRS[dimension]) for _ in range(n_pos)]
        pos = [p for p, _ in pairs]
        neg = [q for _, q in pairs]       # matched one-to-one
    return pos, neg


def generate_scenarios(n_per_dimension: int = 50, seed: int = 0,
                       dimensions: list[str] | None = None) -> list[Scenario]:
    """Generate a balanced, randomized set of dilemmas.

    n_per_dimension dilemmas for each character dimension. Context
    attributes (lane, passengers, signals) are drawn independently.
    """
    rng = random.Random(seed)
    dims = dimensions or CHARACTER_DIMENSIONS
    out: list[Scenario] = []
    sid = 0
    for dim in dims:
        for _ in range(n_per_dimension):
            pos, neg = _groups(rng, dim)
            pos_lane = rng.choice(["ahead", "other"])
            # 1 dilemma in 4: one group are passengers (car hits a barrier)
            passengers = rng.choice(["pos", "neg"]) if rng.random() < 0.25 else None
            signals: dict = {}
            if passengers is None and rng.random() < 0.5:
                # one group crosses on green, the other on red
                legal = rng.choice(["pos", "neg"])
                signals = {legal: "green", ("neg" if legal == "pos" else "pos"): "red"}
            out.append(Scenario(sid, dim, pos, neg, pos_lane, passengers, signals))
            sid += 1
    rng.shuffle(out)
    return out


# --- Text rendering -------------------------------------------------------
def describe_group(chars: list[str]) -> str:
    """['boy','boy','girl'] -> '2 boys and 1 girl'."""
    counts: dict[str, int] = {}
    for c in chars:
        counts[c] = counts.get(c, 0) + 1
    parts = []
    for c, n in counts.items():
        sing, plur = NOUNS[c]
        parts.append(f"{n} {sing if n == 1 else plur}")
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _who(s: Scenario, role: str) -> str:
    """Extra clause about the group's situation (passengers / signal)."""
    group = s.pos_group if role == "pos" else s.neg_group
    are = "is" if len(group) == 1 else "are"
    if s.passengers == role:
        return f", who {are} {'a passenger' if len(group) == 1 else 'passengers'} in the car"
    sig = s.signals.get(role)
    if sig == "green":
        return f", who {are} abiding by the law by crossing on the green signal"
    if sig == "red":
        return f", who {are} flouting the law by crossing on the red signal"
    return f", who {are} crossing the street"


def killed_lane(action: str) -> str:
    return "ahead" if action == "stay" else "other"


def group_text(s: Scenario, lane: str, with_situation: bool = True) -> str:
    """'3 boys, who are crossing the street' (or just '3 boys')."""
    text = describe_group(s.group(lane))
    return text + (_who(s, s.role(lane)) if with_situation else "")


def move_text(s: Scenario, action: str) -> str:
    """What the car physically does for this action."""
    if s.passengers == s.role(killed_lane(action)):
        return ("continue ahead and crash into a concrete barrier" if action == "stay"
                else "swerve and crash into a concrete barrier")
    return ("continue ahead and drive through a pedestrian crossing ahead" if action == "stay"
            else "swerve and drive through a pedestrian crossing in the other lane")


def describe_case(s: Scenario, action: str, mention_spared: bool = True) -> str:
    """Describe the outcome of one action: 'stay' or 'swerve'.
    mention_spared=False gives the shorter 'deaths only' description."""
    k = killed_lane(action)
    sp = "other" if k == "ahead" else "ahead"
    text = (f"The car will {move_text(s, action)}. This will result in the death of "
            f"{group_text(s, k)}.")
    if mention_spared:
        text += f" It will spare {group_text(s, sp)}."
    return text
