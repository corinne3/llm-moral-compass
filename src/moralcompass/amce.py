"""Average Marginal Component Effect (AMCE) with bootstrap confidence intervals.

For a binary attribute (e.g. young vs elderly), the AMCE is:

    AMCE = P(a group is spared | it has the level 'young')
         - P(a group is spared | it has the level 'elderly')

In a forced choice between a 'young' group and an 'elderly' group, the
young group is spared with probability p and the elderly one with 1 - p,
so for one dilemma the effect is p - (1 - p) = 2p - 1, between -1 and +1.
The AMCE is the mean of 2p - 1 over the dilemmas that test this attribute.
Because all other attributes are randomized independently, this simple
mean is an unbiased (causal) estimate: that is the conjoint design.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import CHARACTER_DIMENSIONS, DIMENSIONS
from .scenarios import Scenario


def scenario_effects(raw: pd.DataFrame, scenarios: list[Scenario],
                     group_cols: list[str] | None = None) -> pd.DataFrame:
    """One row per (dilemma, dimension) with its effect 2p - 1.

    p_stay is first averaged over orders (cancels position bias) and,
    unless 'template' is in group_cols, over templates too.
    """
    group_cols = list(group_cols or [])
    keys = ["scenario_id"] + group_cols
    p = raw.groupby(keys, as_index=False)["p_stay"].mean()
    info = {s.id: s for s in scenarios}
    rows = []
    for rec in p.itertuples(index=False):
        s = info[rec.scenario_id]
        p_stay = rec.p_stay
        # probability that the 'pos' group survives
        p_pos = p_stay if s.pos_lane == "other" else 1 - p_stay
        extra = {c: getattr(rec, c) for c in group_cols}
        rows.append({"scenario_id": s.id, "dimension": s.dimension,
                     "effect": 2 * p_pos - 1, **extra})
        # inaction: the group in the other lane survives if the car stays
        rows.append({"scenario_id": s.id, "dimension": "intervention",
                     "effect": 2 * p_stay - 1, **extra})
        if s.passengers is not None:
            p_ped = p_pos if s.passengers == "neg" else 1 - p_pos
            rows.append({"scenario_id": s.id, "dimension": "relation",
                         "effect": 2 * p_ped - 1, **extra})
        if s.signals:
            legal = next(k for k, v in s.signals.items() if v == "green")
            p_legal = p_pos if legal == "pos" else 1 - p_pos
            rows.append({"scenario_id": s.id, "dimension": "law",
                         "effect": 2 * p_legal - 1, **extra})
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, n_boot: int = 1000, seed: int = 0,
                   level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile bootstrap confidence interval."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(values.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def amce_table(effects: pd.DataFrame, group_cols: list[str] | None = None,
               n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """AMCE + 95% CI for each dimension (and each group, if given)."""
    group_cols = list(group_cols or [])
    rows = []
    for key, g in effects.groupby(group_cols + ["dimension"]):
        key = key if isinstance(key, tuple) else (key,)
        mean, lo, hi = bootstrap_mean(g["effect"].values, n_boot, seed)
        rows.append({**dict(zip(group_cols + ["dimension"], key)),
                     "amce": mean, "ci_low": lo, "ci_high": hi, "n": len(g)})
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    out["_o"] = out["dimension"].map(order)
    return out.sort_values(group_cols + ["_o"]).drop(columns="_o").reset_index(drop=True)


def amce_vector(table: pd.DataFrame) -> np.ndarray:
    """The 9 AMCEs in the fixed DIMENSIONS order."""
    s = table.set_index("dimension")["amce"]
    return np.array([s.get(d, np.nan) for d in DIMENSIONS], dtype=float)


def paired_delta(base: pd.DataFrame, cond: pd.DataFrame, n_boot: int = 1000,
                 seed: int = 0) -> pd.DataFrame:
    """Change of AMCE between two conditions measured on the SAME dilemmas.

    Pairing (same dilemma, two conditions) removes the dilemma-to-dilemma
    variability, so small shifts become detectable with fewer dilemmas.
    """
    m = base.merge(cond, on=["scenario_id", "dimension"], suffixes=("_base", "_cond"))
    rows = []
    for d, g in m.groupby("dimension"):
        diff = g["effect_cond"].values - g["effect_base"].values
        mean, lo, hi = bootstrap_mean(diff, n_boot, seed)
        rows.append({"dimension": d, "delta": mean, "ci_low": lo, "ci_high": hi,
                     "significant": bool(lo > 0 or hi < 0), "n": len(g)})
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    return out.sort_values("dimension", key=lambda c: c.map(order)).reset_index(drop=True)


def template_sensitivity(raw: pd.DataFrame, scenarios: list[Scenario]) -> pd.DataFrame:
    """AMCE computed separately for each prompt wording, and their spread."""
    eff = scenario_effects(raw, scenarios, group_cols=["template"])
    tab = amce_table(eff, group_cols=["template"], n_boot=200)
    wide = tab.pivot(index="dimension", columns="template", values="amce")
    wide.columns = [f"template_{c}" for c in wide.columns]
    wide["spread_sd"] = wide.std(axis=1)
    return wide.reindex([d for d in DIMENSIONS if d in wide.index])


def amce_by_template(raw: pd.DataFrame, scenarios: list[Scenario], n_boot: int = 1000) -> pd.DataFrame:
    """AMCE and 95% CI for each wording / instrument separately."""
    eff = scenario_effects(raw, scenarios, group_cols=["template"])
    return amce_table(eff, group_cols=["template"], n_boot=n_boot)


def robustness(by_template: pd.DataFrame, templates: list[int]) -> pd.DataFrame:
    """A preference is called ROBUST when, on every listed wording, its
    95% CI excludes 0 with the same sign. Otherwise it depends on how the
    question is asked."""
    rows = []
    t = by_template[by_template.template.isin(templates)]
    for d, g in t.groupby("dimension"):
        pos = (g.ci_low > 0).all()
        neg = (g.ci_high < 0).all()
        rows.append({"dimension": d, "min": g.amce.min(), "max": g.amce.max(),
                     "verdict": "robust +" if pos else "robust -" if neg else "depends on wording"})
    out = pd.DataFrame(rows)
    order = {d: i for i, d in enumerate(DIMENSIONS)}
    return out.sort_values("dimension", key=lambda c: c.map(order)).reset_index(drop=True)


__all__ = ["amce_by_template", "robustness", "scenario_effects", "bootstrap_mean", "amce_table", "amce_vector",
           "paired_delta", "template_sensitivity", "CHARACTER_DIMENSIONS"]
