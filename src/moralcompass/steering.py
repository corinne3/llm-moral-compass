"""Activation steering: push the model's internal state toward an emotion
and measure how its moral choices move.

Four methods, all injected at the same layer and with the SAME size of
perturbation in the residual stream (a fair comparison):

- direction:      add the emotion direction (difference of means)
- neurons:        add values to the top-k emotion neurons of the MLP
                  (the idea of the Hemicycle project: "boost the fear neurons")
- random:         add a random direction  -> control
- random_neurons: boost k random neurons  -> control

Size = alpha x (mean norm of the residual stream at that layer).
alpha = 0.1 means a push of 10% of the typical activation size.
If a 'direction' effect is not larger than the 'random' effect, the
emotion is not what moved the model: any perturbation would.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .amce import paired_delta, scenario_effects
from .interventions import AddToNeurons, AddToResidual
from .measure import run_dilemmas

METHODS = ["direction", "neurons", "random", "random_neurons"]


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("|".join(map(str, keys)).encode()))


def build_intervention(model, method: str, emotion: str, layer: int, alpha: float,
                       steering_data: dict, seed: int = 0):
    """Return the intervention object for one condition (or None if alpha = 0)."""
    if alpha == 0:
        return None
    target = alpha * steering_data["resid_norm"][layer]
    if method == "direction":
        unit = steering_data["directions"][layer][emotion]
        return AddToResidual(layer, unit * target)
    if method == "random":
        v = _rng("random", emotion, layer, seed).normal(size=model.d_model)
        return AddToResidual(layer, v / np.linalg.norm(v) * target)
    if method in ("neurons", "random_neurons"):
        if method == "neurons":
            n = steering_data["neurons"][layer][emotion]
            idx, sign, std = n["indices"], np.sign(n["cohen_d"]), n["std"]
        else:
            k = len(steering_data["neurons"][layer][emotion]["indices"])
            r = _rng("rneurons", emotion, layer, seed)
            idx = r.choice(model.d_mlp, size=k, replace=False)
            sign = r.choice([-1.0, 1.0], size=k)
            std = steering_data["neuron_std"][layer][idx]
        base = sign * std                                   # 1 std per neuron
        u = model.neuron_residual_effect(layer, idx, base)  # its residual footprint
        scale = target / max(np.linalg.norm(u), 1e-12)      # match the perturbation size
        return AddToNeurons(layer, np.asarray(idx), base * scale)
    raise ValueError(method)


def dose_response(model, scenarios, steering_data: dict, layer: int,
                  emotions=("fear", "anger", "hope"), methods=METHODS,
                  alphas=(0.0, 0.1, 0.25, 0.5), templates=(0,), seed: int = 0,
                  progress=None):
    """Measure the dilemmas under every (method, emotion, alpha).

    Returns (raw, deltas):
      raw    all choice probabilities
      deltas change of each AMCE vs alpha = 0, paired on the same dilemmas
    """
    base_raw = run_dilemmas(model, scenarios, templates=templates, condition="alpha0")
    base_raw = base_raw.assign(method="none", emotion="none", alpha=0.0)
    base_eff = scenario_effects(base_raw, scenarios)
    raws, deltas = [base_raw], []
    jobs = [(m, e, a) for m in methods for e in emotions for a in alphas if a > 0]
    for i, (m, e, a) in enumerate(jobs):
        iv = build_intervention(model, m, e, layer, a, steering_data, seed)
        with model.steer(iv):
            r = run_dilemmas(model, scenarios, templates=templates, condition=f"{m}:{e}:{a}")
        r = r.assign(method=m, emotion=e, alpha=a)
        raws.append(r)
        d = paired_delta(base_eff, scenario_effects(r, scenarios))
        deltas.append(d.assign(method=m, emotion=e, alpha=a, mean_mass=r["mass"].mean(),
                               # does the emotion change the preference for answer "1"?
                               position_bias=r["p_case1"].mean() - 0.5,
                               base_position_bias=base_raw["p_case1"].mean() - 0.5))
        if progress:
            progress(i + 1, len(jobs), m, e, a)
    return pd.concat(raws, ignore_index=True), pd.concat(deltas, ignore_index=True)


def steered_samples(model, steering_data: dict, layer: int, alpha: float,
                    question: str = "How do you feel right now? Answer in two sentences.",
                    emotions=("fear", "anger", "hope")) -> dict:
    """Let the steered model write a little: does it *sound* afraid, angry, hopeful?"""
    msgs = [{"role": "user", "content": question}]
    out = {"no steering": model.generate(msgs)}
    for e in emotions:
        iv = build_intervention(model, "direction", e, layer, alpha, steering_data)
        with model.steer(iv):
            out[f"{e} (direction, alpha={alpha})"] = model.generate(msgs)
    return out


# ---------------------------------------------------------------------------
# Is a steering effect real? Check it on every wording / instrument.
# ---------------------------------------------------------------------------
def deltas_by_template(raw: pd.DataFrame, scenarios, n_boot: int = 1000) -> pd.DataFrame:
    """Paired change of each AMCE vs alpha = 0, separately for each wording."""
    base = raw[raw.alpha == 0]
    be = scenario_effects(base, scenarios, group_cols=["template"])
    rows = []
    for (m, e, a), g in raw[raw.alpha > 0].groupby(["method", "emotion", "alpha"]):
        ce = scenario_effects(g, scenarios, group_cols=["template"])
        for t in sorted(g.template.unique()):
            d = paired_delta(be[be.template == t].drop(columns="template"),
                             ce[ce.template == t].drop(columns="template"), n_boot)
            rows.append(d.assign(method=m, emotion=e, alpha=a, template=t))
    return pd.concat(rows, ignore_index=True)


def consistency(dbt: pd.DataFrame) -> pd.DataFrame:
    """One row per (method, emotion, alpha, dimension).

    - 'sig. on all wordings': the 95% CI excludes 0, same sign, on every wording
    - 'same sign': the change goes the same way on every wording (weaker)
    - 'inconsistent': it depends on the wording -> an instrument artefact
    The random controls give the false-positive rate of each verdict.
    """
    rows = []
    for (m, e, a, dim), g in dbt.groupby(["method", "emotion", "alpha", "dimension"]):
        same = bool((g.delta > 0).all() or (g.delta < 0).all())
        allsig = bool((g.ci_low > 0).all() or (g.ci_high < 0).all())
        rows.append({"method": m, "emotion": e, "alpha": a, "dimension": dim,
                     "mean_delta": g.delta.mean(), "min_delta": g.delta.min(),
                     "max_delta": g.delta.max(), "n_templates": len(g),
                     "verdict": ("sig. on all wordings" if allsig else
                                 "same sign" if same else "inconsistent")})
    return pd.DataFrame(rows)


def consistency_summary(cons: pd.DataFrame) -> pd.DataFrame:
    """How many of the 9 dimensions move consistently, per method/emotion/alpha."""
    return (cons.assign(sig_all=cons.verdict.eq("sig. on all wordings"),
                        same_sign=cons.verdict.ne("inconsistent"))
                .groupby(["method", "emotion", "alpha"])[["sig_all", "same_sign"]].sum()
                .astype(int))
