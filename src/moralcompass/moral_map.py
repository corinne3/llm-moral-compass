"""Place models among countries on a 'moral map'.

Each country (and each model) is a point in a 9-dimensional space: its
9 AMCEs. To draw it in 2D we use PCA, fitted on the countries only:
- standardize each dimension with the countries' mean and std
  (otherwise dimensions with large spread dominate),
- PCA finds the 2 axes along which countries differ the most,
- models are projected onto those same axes.
Distances are computed in the full 9-D space, in raw AMCE units, like
the Euclidean distance used in the LLM Moral Machine literature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.decomposition import PCA

from . import DIMENSIONS


def standardize(human: pd.DataFrame):
    mu = np.array(human[DIMENSIONS].mean(), dtype=float)
    sd = np.array(human[DIMENSIONS].std(), dtype=float)
    sd[sd == 0] = 1.0
    return mu, sd


def fit_map(human: pd.DataFrame, n_components: int = 2):
    mu, sd = standardize(human)
    pca = PCA(n_components=n_components).fit((human[DIMENSIONS].values - mu) / sd)
    return {"mu": mu, "sd": sd, "pca": pca}


def project(m: dict, vectors: np.ndarray) -> np.ndarray:
    v = np.atleast_2d(vectors)
    return m["pca"].transform((v - m["mu"]) / m["sd"])


def cluster_countries(human: pd.DataFrame, k: int = 3) -> pd.Series:
    """Ward hierarchical clustering on standardized AMCEs.
    Awad et al. found 3 cultural clusters (Western, Eastern, Southern)."""
    mu, sd = standardize(human)
    Z = linkage((human[DIMENSIONS].values - mu) / sd, method="ward")
    return pd.Series(fcluster(Z, k, criterion="maxclust"), index=human.index, name="cluster")


def distances(human: pd.DataFrame, vec: np.ndarray) -> pd.Series:
    """Euclidean distance (raw AMCE units) from a model to every country."""
    d = np.sqrt(((human[DIMENSIONS].values - vec) ** 2).sum(axis=1))
    return pd.Series(d, index=human.index, name="distance").sort_values()


def nearest_cluster(human: pd.DataFrame, clusters: pd.Series, vec: np.ndarray) -> dict:
    out = {}
    for c in sorted(clusters.unique()):
        centroid = human.loc[clusters == c, DIMENSIONS].mean().values
        out[int(c)] = float(np.sqrt(((centroid - vec) ** 2).sum()))
    return out


def alignment_rank(human: pd.DataFrame, country: str, vec: np.ndarray) -> dict:
    """Cultural prompting check: among all countries, how close is the
    target country to the model's answers? rank 1 = the closest."""
    d = distances(human, vec)
    rank = int(list(d.index).index(country)) + 1
    return {"country": country, "rank": rank, "n": len(d),
            "percentile": 1 - (rank - 1) / (len(d) - 1), "distance": float(d[country])}


def plot_map(human: pd.DataFrame, models: dict, path: str, clusters: pd.Series | None = None,
             label_countries: list[str] | None = None, title: str = "Moral map"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = fit_map(human)
    xy = project(m, human[DIMENSIONS].values)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = None if clusters is None else clusters.values
    ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="Set2", s=22, alpha=0.7,
               edgecolor="none", label="countries (humans)")
    for name in (label_countries or []):
        if name in human.index:
            i = list(human.index).index(name)
            ax.annotate(name, xy[i], fontsize=7, color="#555", xytext=(3, 3),
                        textcoords="offset points")
    glob = project(m, human[DIMENSIONS].mean().values)[0]
    ax.scatter(*glob, marker="D", s=70, color="black", label="human average")
    markers = "o^sPX*v<>"
    for i, (name, vec) in enumerate(models.items()):
        p = project(m, vec)[0]
        ax.scatter(*p, marker=markers[i % len(markers)], s=110, edgecolor="black",
                   linewidth=0.8, label=name, zorder=5)
    ev = m["pca"].explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of country variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"explained_variance": ev.tolist(),
            "loadings": pd.DataFrame(m["pca"].components_.T, index=DIMENSIONS,
                                     columns=["PC1", "PC2"])}
