"""Run dilemmas through a model and collect choice probabilities.

Two instruments, both read from probabilities (no sampling):

1. Numbered answer (templates 0-4): at the position where the answer
   starts, the probability of the token "1" and of the token "2".
2. Answer by name (template 5, cloze scoring): the assistant's answer is
   started with "The group that should be spared is:" and we compute the
   probability of each full group name, e.g. " 2 dogs." vs
   " 1 male doctor and 1 female athlete.", token by token (chain rule).
   No number to bind to the wrong case.
   Template 6 = template 5 CALIBRATED: we also compute each name's
   probability without the dilemma (just the list) and subtract it:
       score = log P(name | dilemma) - log P(name | list only)
   This removes the model's a-priori taste for some strings (short words,
   small numbers, frequent words): pointwise mutual information (PMI).

Both give, per prompt, p_case1 = probability of the FIRST listed option,
so all the downstream statistics are identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prompts import (CLOZE_CAL, CLOZE_PREFIX, CLOZE_TEMPLATES, build_messages,
                      cloze_options, cloze_prior_messages)
from .scenarios import Scenario


def run_dilemmas(model, scenarios: list[Scenario], templates=(0, 1, 2), orders=("A", "B"),
                 system: str | None = None, context: list[dict] | None = None,
                 condition: str = "baseline") -> pd.DataFrame:
    """One row per (dilemma, template, order).

    Columns:
      p_case1  probability of answering "1" (renormalized over 1/2)
      mass     total probability of "1"+"2" before renormalizing
      p_stay   probability that the car STAYS on course
               (= p_case1 in order A, = 1 - p_case1 in order B)
    """
    num_jobs, num_meta, clz_jobs, clz_pairs, clz_meta, clz_prior = [], [], [], [], [], []
    for s in scenarios:
        for t in templates:
            for o in orders:
                msgs = build_messages(s, t, o, system=system, context=context)
                if t in CLOZE_TEMPLATES:
                    clz_jobs.append(msgs)
                    clz_pairs.append(cloze_options(s, o))
                    clz_meta.append((s.id, s.dimension, t, o))
                    clz_prior.append(cloze_prior_messages(s, o) if t == CLOZE_CAL else None)
                else:
                    num_jobs.append(msgs)
                    num_meta.append((s.id, s.dimension, t, o))
    cols = ["scenario_id", "dimension", "template", "order"]
    parts = []
    if num_jobs:
        probs = model.option_probs(num_jobs, options=("1", "2"))
        d = pd.DataFrame(num_meta, columns=cols)
        d["p_case1"], d["mass"] = probs[:, 0], probs[:, -1]
        parts.append(d)
    if clz_jobs:
        lp = model.cloze_logprobs(clz_jobs, clz_pairs, CLOZE_PREFIX)
        score = lp.copy()
        cal = [i for i, pr in enumerate(clz_prior) if pr is not None]
        if cal:   # template 6: subtract the a-priori log-probability of each name
            prior = model.cloze_logprobs([clz_prior[i] for i in cal],
                                         [clz_pairs[i] for i in cal], CLOZE_PREFIX)
            score[cal] = lp[cal] - prior
        d = pd.DataFrame(clz_meta, columns=cols)
        d["p_case1"] = 1 / (1 + np.exp(score[:, 1] - score[:, 0]))
        d["mass"] = np.exp(lp[:, 0]) + np.exp(lp[:, 1])   # prob. of writing exactly one of the 2 names
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["p_stay"] = np.where(df["order"] == "A", df["p_case1"], 1 - df["p_case1"])
    df["condition"] = condition
    return df


def scenario_table(scenarios: list[Scenario]) -> pd.DataFrame:
    """Design matrix: one row per dilemma with its attributes."""
    rows = []
    for s in scenarios:
        rows.append({
            "scenario_id": s.id,
            "dimension": s.dimension,
            "pos_lane": s.pos_lane,
            "passengers": s.passengers or "none",
            "legal_side": next((k for k, v in s.signals.items() if v == "green"), "none"),
            "n_pos": len(s.pos_group),
            "n_neg": len(s.neg_group),
        })
    return pd.DataFrame(rows)


def diagnostics(raw: pd.DataFrame) -> dict:
    """Quality checks of the measurement itself (not of the model's morals)."""
    by = raw.groupby(["scenario_id", "template"])
    pa = by.apply(lambda g: g.loc[g["order"] == "A", "p_stay"].mean(), include_groups=False)
    pb = by.apply(lambda g: g.loc[g["order"] == "B", "p_stay"].mean(), include_groups=False)
    both = pd.concat([pa, pb], axis=1, keys=["A", "B"]).dropna()
    consistent = ((both["A"] > 0.5) == (both["B"] > 0.5)).mean() if len(both) else float("nan")
    return {
        # > 0: the model prefers whatever is labelled "1"
        "position_bias": float(raw["p_case1"].mean() - 0.5),
        # share of dilemmas where the decision survives swapping the order
        "order_consistency": float(consistent),
        # how much probability went to "1"/"2" at all (format validity)
        "mean_mass": float(raw["mass"].mean()),
        "min_mass": float(raw["mass"].min()),
        "n_prompts": int(len(raw)),
        # What a single-order study would have concluded about "inaction"
        # (stay on course). Compare with the counterbalanced AMCE.
        "naive_inaction_order_A_only": float(2 * raw.loc[raw.order == "A", "p_stay"].mean() - 1),
        "naive_inaction_order_B_only": float(2 * raw.loc[raw.order == "B", "p_stay"].mean() - 1),
    }


def diagnostics_by_template(raw: pd.DataFrame) -> pd.DataFrame:
    """The same quality checks, separately for each wording."""
    rows = []
    for t, g in raw.groupby("template"):
        d = diagnostics(g)
        rows.append({"template": t, "position_bias": d["position_bias"],
                     "order_consistency": d["order_consistency"], "mean_mass": d["mean_mass"]})
    return pd.DataFrame(rows).set_index("template").round(3)
