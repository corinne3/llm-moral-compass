"""Contrastive sentences to find emotion directions inside the model.

Same 20 everyday situations (nothing about traffic or morality), written
in matched frames: the four classes differ ONLY by the emotion words, and
the neutral frames have the SAME LENGTH as the emotional ones (a first
version had shorter neutral sentences: the 'emotion' directions all turned
out to be one 'sentence length' direction, cosine 0.999).
So when we average activations per class and subtract, what remains is
(mostly) the emotion. Each sentence remembers its situation, so probes
can be tested on situations never seen in training (no leakage).
"""

from __future__ import annotations

EMOTIONS = ["fear", "anger", "hope"]
CLASSES = ["neutral"] + EMOTIONS

SITUATIONS = [
    "the exam results next week", "my new job", "the dinner with my family",
    "the weather this weekend", "the renovation of my kitchen", "the letter from the bank",
    "my sister's wedding", "moving to a new city", "the meeting with my manager",
    "the harvest in my garden", "the new park in my town", "my dog's training",
    "the change in the train timetable", "the phone call from my cousin",
    "the price of groceries", "the book I am writing", "my neighbour's party",
    "the result of the football match", "the update of my computer", "the school trip of my son",
]

FRAMES = {
    "neutral": [   # same length as the emotional frames: only the emotion words differ
        "I am thinking about {s}.",
        "Today I wrote a note about {s}, and I put it on my desk.",
        "This is what I know about {s}: it is happening soon, and I wrote down the date.",
        "When I consider {s}, I list the facts and check what comes next.",
    ],
    "fear": [
        "I am terrified about {s}.",
        "Today I wrote a note about {s}, and I felt scared and anxious.",
        "This is what I know about {s}: it is happening soon, and I am filled with dread.",
        "When I consider {s}, I panic and feel afraid of what could go wrong.",
    ],
    "anger": [
        "I am furious about {s}.",
        "Today I wrote a note about {s}, and I felt angry and outraged.",
        "This is what I know about {s}: it is happening soon, and I am livid about it.",
        "When I consider {s}, I am fuming with rage at how unfair it is.",
    ],
    "hope": [
        "I am hopeful about {s}.",
        "Today I wrote a note about {s}, and I felt optimistic and confident.",
        "This is what I know about {s}: it is happening soon, and I am looking forward to it.",
        "When I consider {s}, I believe things will turn out bright and better.",
    ],
}


def build_dataset() -> list[dict]:
    """List of {text, label, situation}."""
    rows = []
    for si, s in enumerate(SITUATIONS):
        for label, frames in FRAMES.items():
            for f in frames:
                rows.append({"text": f.format(s=s), "label": label, "situation": si})
    return rows
