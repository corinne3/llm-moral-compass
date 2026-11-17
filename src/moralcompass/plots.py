"""All figures. Matplotlib only, saved as PNG."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import DIMENSION_LABELS, DIMENSIONS  # noqa: E402

PALETTE = ["#2a6fdb", "#e0703a", "#2e9d6a", "#9b59b6", "#c0392b", "#7f8c8d"]


def plot_amce(tables: dict, path: str, human: np.ndarray | None = None,
              title: str = "Moral preferences (AMCE, 95% CI)"):
    """Horizontal bars, one group of bars per dimension."""
    names = list(tables)
    y = np.arange(len(DIMENSIONS))
    h = 0.8 / (len(names) + (human is not None))
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(DIMENSIONS) + 1.5))
    for i, n in enumerate(names):
        t = tables[n].set_index("dimension").reindex(DIMENSIONS)
        err = np.vstack([t["amce"] - t["ci_low"], t["ci_high"] - t["amce"]])
        ax.barh(y + i * h, t["amce"], height=h, xerr=err, color=PALETTE[i % len(PALETTE)],
                label=n, capsize=2, error_kw={"lw": 0.8})
    if human is not None:
        ax.barh(y + len(names) * h, human, height=h, color="#333", label="humans (country mean)")
    ax.set_yticks(y + 0.4 - h / 2)
    ax.set_yticklabels([DIMENSION_LABELS[d] for d in DIMENSIONS])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("ΔP (probability of sparing the first-named group minus the other)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_probes(scores: pd.DataFrame, path: str, chosen_layer: int | None = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    styles = {"logistic": "-", "mean_diff": "--"}
    for i, emo in enumerate(["fear", "anger", "hope"]):
        for probe, ls in styles.items():
            s = scores[(scores.target == emo) & (scores.probe == probe)].sort_values("layer")
            ax.plot(s.layer, s.accuracy, ls, color=PALETTE[i], label=f"{emo} vs neutral ({probe})")
    s = scores[scores.target == "4-class"].sort_values("layer")
    ax.plot(s.layer, s.accuracy, ":", color="black", label="4 classes (logistic)")
    ax.axhline(0.5, color="grey", lw=0.6)
    ax.axhline(0.25, color="grey", lw=0.6, ls=":")
    if chosen_layer is not None:
        ax.axvline(chosen_layer, color="red", lw=1, alpha=0.6, label=f"steering layer {chosen_layer}")
    ax.set_xlabel("layer")
    ax.set_ylabel("held-out accuracy (new situations)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Where are emotions linearly readable?")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dose_response(deltas: pd.DataFrame, emotion: str, path: str):
    """3x3 grid: one panel per dimension; one line per method."""
    d = deltas[deltas.emotion == emotion]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True)
    methods = list(dict.fromkeys(d.method))
    for ax, dim in zip(axes.flat, DIMENSIONS):
        for i, m in enumerate(methods):
            s = d[(d.method == m) & (d.dimension == dim)].sort_values("alpha")
            x = np.r_[0, s.alpha.values]
            ax.plot(x, np.r_[0, s.delta.values], "o-", color=PALETTE[i], ms=3, label=m)
            ax.fill_between(x, np.r_[0, s.ci_low.values], np.r_[0, s.ci_high.values],
                            color=PALETTE[i], alpha=0.15)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(DIMENSION_LABELS[dim], fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes[-1]:
        ax.set_xlabel("alpha (push size / activation size)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Δ AMCE vs no steering")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Steering toward '{emotion}': change of each moral preference")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_delta_heatmap(deltas: pd.DataFrame, path: str, row_col: str = "condition",
                       title: str = "Change of AMCE vs baseline"):
    """Rows = conditions, columns = dimensions; * = 95% CI excludes 0."""
    piv = deltas.pivot(index=row_col, columns="dimension", values="delta").reindex(columns=DIMENSIONS)
    sig = deltas.pivot(index=row_col, columns="dimension", values="significant").reindex(columns=DIMENSIONS)
    vmax = np.nanmax(np.abs(piv.values)) or 1.0
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(piv) + 2))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                star = "*" if bool(sig.values[i, j]) else ""
                ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(DIMENSIONS)))
    ax.set_xticklabels([DIMENSION_LABELS[d] for d in DIMENSIONS], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Δ AMCE")
    ax.set_title(title + "   (* = 95% CI excludes 0)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_side_effects(df: pd.DataFrame, path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, (key, g) in enumerate(df.groupby(["method", "emotion"])):
        g = g.sort_values("alpha")
        lab = f"{key[0]} / {key[1]}"
        axes[0].plot(g.alpha, g.accuracy, "o-", ms=3, color=PALETTE[i % 6], label=lab)
        axes[1].plot(g.alpha, g.perplexity, "o-", ms=3, color=PALETTE[i % 6], label=lab)
    axes[0].set_ylabel("accuracy on 30 easy questions")
    axes[1].set_ylabel("perplexity on neutral text")
    for ax in axes:
        ax.set_xlabel("alpha")
        ax.grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle("Side effects: does steering damage the model?")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
