"""LLM Moral Compass.

Measure the moral preferences of language models with the Moral Machine
design, place them on a map next to 130 countries, and test whether
prompting or activation steering (fear, anger, hope) moves their choices.
"""

__version__ = "0.1.0"

# The nine Moral Machine dimensions, in a fixed order used everywhere.
# Each name is the *preference* measured, so a positive AMCE means
# "the model spares the first-named group more often".
DIMENSIONS = [
    "species",        # spare humans (vs pets)
    "number",         # spare more characters (vs fewer)
    "age",            # spare the young (vs the elderly)
    "law",            # spare law-abiding pedestrians (vs jaywalkers)
    "social_status",  # spare higher status (vs lower status)
    "fitness",        # spare the fit (vs the large)
    "gender",         # spare females (vs males)
    "relation",       # spare pedestrians (vs passengers)
    "intervention",   # prefer inaction: stay on course (vs swerve)
]

DIMENSION_LABELS = {
    "species": "Humans over pets",
    "number": "More lives over fewer",
    "age": "Young over elderly",
    "law": "Lawful over jaywalking",
    "social_status": "Higher over lower status",
    "fitness": "Fit over large",
    "gender": "Females over males",
    "relation": "Pedestrians over passengers",
    "intervention": "Inaction over swerving",
}

# The six dimensions that are tested by choosing *who* is in each group.
CHARACTER_DIMENSIONS = ["species", "number", "age", "social_status", "fitness", "gender"]
