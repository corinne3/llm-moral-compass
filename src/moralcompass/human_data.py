"""Load the human reference: country-level AMCEs from Awad et al. (Nature, 2018).

The Moral Machine team published one AMCE per country and per dimension
(about 130 countries with enough respondents). See the README for how to
download the file. This loader accepts two formats:

1. The original wide file (one row per country, one "...: Estimates"
   column per dimension, plus standard-error columns that we ignore).
2. A simple tidy file with columns: country, dimension, amce
   (dimension names as in moralcompass.DIMENSIONS).

Column names in the original file are matched by keywords, and signs are
normalized so that every dimension is expressed as the preference named
in DIMENSIONS (e.g. "spare the young"). Any flip is reported.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import DIMENSIONS

# Checked in this order; the first keyword match wins.
COLUMN_KEYWORDS = [
    ("intervention", ["omission", "commission", "intervention", "inaction"]),
    ("relation", ["passenger", "pedestrian"]),
    ("law", ["illegal", "legal", "law"]),
    ("gender", ["gender", "female"]),
    ("fitness", ["fitness"]),
    ("social_status", ["social", "status"]),
    ("age", ["age", "elderly", "young"]),
    ("number", ["no. characters", "number", "characters", "utilitarian"]),
    ("species", ["species", "pets", "humans"]),
]

_IGNORE = re.compile(r"(\bse\b|\bsd\b|std|stderr|\bci\b|lower|upper|\bn\b)", re.I)


def _map_columns(columns) -> dict:
    estimate_cols = [c for c in columns if "estimate" in c.lower()]
    candidates = estimate_cols or [c for c in columns if not _IGNORE.search(c)]
    mapping = {}
    for c in candidates:
        low = c.lower()
        for dim, words in COLUMN_KEYWORDS:
            if dim not in mapping.values() and any(w in low for w in words):
                mapping[c] = dim
                break
    return mapping


def load_human_amce(path: str) -> tuple[pd.DataFrame, dict]:
    """Return (table countries x 9 dimensions, info dict)."""
    raw = pd.read_csv(path)
    info: dict = {"path": path, "flipped": [], "mapping": {}}
    cols_low = [c.lower() for c in raw.columns]
    if {"country", "dimension", "amce"} <= set(cols_low):
        raw.columns = cols_low
        table = raw.pivot(index="country", columns="dimension", values="amce")
    else:
        country_col = next((c for c in raw.columns
                            if c.lower() in ("country", "countries", "iso3", "code", "unnamed: 0")),
                           raw.columns[0])
        mapping = _map_columns([c for c in raw.columns if c != country_col])
        info["mapping"] = mapping
        table = raw.set_index(country_col)[list(mapping)].rename(columns=mapping)
        table.index.name = "country"
    missing = [d for d in DIMENSIONS if d not in table.columns]
    if missing:
        raise ValueError(f"Dimensions not found in {path}: {missing}. "
                         f"Columns were: {list(raw.columns)}")
    table = table[DIMENSIONS].astype(float).dropna()
    # Nature 2018, Fig. 2: all nine global preferences are positive.
    for d in DIMENSIONS:
        if table[d].mean() < 0:
            table[d] = -table[d]
            info["flipped"].append(d)
    info["n_countries"] = int(len(table))
    return table, info


def human_reference(table: pd.DataFrame) -> np.ndarray:
    """Average over countries: an approximation of the global human profile
    (the paper's global estimate pools all respondents, so it differs a bit)."""
    return table[DIMENSIONS].mean().values


CLUSTER_NAMES = {0: "Western", 1: "Eastern", 2: "Southern"}


def load_cluster_map(path: str) -> pd.Series:
    """Official cultural clusters of Awad et al. (country_cluster_map.csv on OSF):
    ISO3 -> 'Western' / 'Eastern' / 'Southern' (0 / 1 / 2 in the file)."""
    m = pd.read_csv(path)
    return m.set_index("ISO3")["Cluster"].map(CLUSTER_NAMES).rename("cluster")


def country_names(path: str) -> dict:
    """ISO3 -> country name, from the same file."""
    m = pd.read_csv(path)
    return dict(zip(m.ISO3, m.Country))
