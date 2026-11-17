"""Where do emotions live inside the model? Linear probes, layer by layer.

Linear representation hypothesis: many concepts are encoded as DIRECTIONS
in the residual stream. If a concept is linearly encoded at layer l, a
simple linear classifier (a 'probe') trained on the activations of
layer l can recognize it on new sentences.

For each layer we compare two ways to find the emotion direction:
- logistic-regression probe (the best linear separator),
- difference of class means (what we will use for steering).
Both are evaluated with GroupKFold: test sentences use situations never
seen during training, so the probe cannot cheat by memorizing topics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .emotions import EMOTIONS


def probe_scores(acts: dict, labels: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validated accuracy per layer, for each emotion vs neutral,
    and for the 4-class problem."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    rows = []
    for layer, X in acts.items():
        for emo in EMOTIONS:
            m = np.isin(labels, ["neutral", emo])
            y = (labels[m] == emo).astype(int)
            Xm, gm = X[m], groups[m]
            acc_lr, acc_md = [], []
            for tr, te in GroupKFold(n_splits).split(Xm, y, gm):
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
                clf.fit(Xm[tr], y[tr])
                acc_lr.append(clf.score(Xm[te], y[te]))
                # difference of means: project, threshold at the midpoint
                mu1, mu0 = Xm[tr][y[tr] == 1].mean(0), Xm[tr][y[tr] == 0].mean(0)
                v = mu1 - mu0
                thr = ((mu1 + mu0) / 2) @ v
                acc_md.append(((Xm[te] @ v > thr).astype(int) == y[te]).mean())
            rows.append({"layer": layer, "target": emo, "probe": "logistic", "accuracy": np.mean(acc_lr)})
            rows.append({"layer": layer, "target": emo, "probe": "mean_diff", "accuracy": np.mean(acc_md)})
        accs = []
        for tr, te in GroupKFold(n_splits).split(X, labels, groups):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
            clf.fit(X[tr], labels[tr])
            accs.append(clf.score(X[te], labels[te]))
        rows.append({"layer": layer, "target": "4-class", "probe": "logistic", "accuracy": np.mean(accs)})
    return pd.DataFrame(rows)


def choose_layer(scores: pd.DataFrame, n_layers: int, tolerance: float = 0.02) -> int:
    """Pick the steering layer: among the middle layers (25%-75% of depth),
    the EARLIEST one whose mean-difference accuracy is within `tolerance`
    of the best. Middle layers hold abstract concepts; steering earlier
    leaves the model more computation to integrate the change."""
    s = scores[(scores.probe == "mean_diff") & (scores.target.isin(EMOTIONS))]
    per_layer = s.groupby("layer")["accuracy"].mean()
    lo, hi = int(0.25 * n_layers), int(np.ceil(0.75 * n_layers))
    mid = per_layer[(per_layer.index >= lo) & (per_layer.index <= hi)]
    if mid.empty:
        mid = per_layer
    best = mid.max()
    return int(mid[mid >= best - tolerance].index.min())


def mean_diff_directions(acts: dict, labels: np.ndarray) -> dict:
    """{layer: {emotion: (unit_vector, raw_norm)}} with v = mean(emotion) - mean(neutral)."""
    labels = np.asarray(labels)
    out = {}
    for layer, X in acts.items():
        mu0 = X[labels == "neutral"].mean(0)
        out[layer] = {}
        for emo in EMOTIONS:
            v = X[labels == emo].mean(0) - mu0
            n = float(np.linalg.norm(v))
            out[layer][emo] = (v / n if n > 0 else v, n)
    return out


def select_neurons(mlp_acts: np.ndarray, labels: np.ndarray, emotion: str, k: int = 32):
    """Top-k MLP neurons by |Cohen's d| between `emotion` and neutral.

    Returns (indices, cohen_d, pooled_std) for the k selected neurons.
    The sign of d says whether the neuron fires MORE (d > 0) or LESS
    for the emotion; steering pushes each neuron in its own direction.
    """
    labels = np.asarray(labels)
    a, b = mlp_acts[labels == emotion], mlp_acts[labels == "neutral"]
    pooled = np.sqrt((a.var(0, ddof=1) + b.var(0, ddof=1)) / 2) + 1e-8
    d = (a.mean(0) - b.mean(0)) / pooled
    idx = np.argsort(-np.abs(d))[:k]
    return idx, d[idx], pooled[idx]
