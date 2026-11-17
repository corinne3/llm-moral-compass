"""Step 8 - The summary figures used in the README (from saved results only).

Writes docs/figures/:
  fig1_position_bias.png   models answering the position, not the dilemma
  fig2_instruments.png     Qwen 7B: the 9 AMCEs under 6 wordings/instruments
  fig3_comprehension.png   Qwen 7B: does it know who dies?
  fig4_steering_controls.png  emotion steering vs random controls
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _path  # noqa: F401,E402
from moralcompass import DIMENSION_LABELS, DIMENSIONS  # noqa: E402
from moralcompass.measure import diagnostics  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", default="results")
p.add_argument("--out", default="docs/figures")
p.add_argument("--big", default="Qwen2.5-7B-Instruct")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
BLUE, ORANGE, GREY, RED = "#2a6fdb", "#e0703a", "#9aa7b4", "#c0392b"

# ---- fig 1: position bias across models (wordings 0-2 only, same for all) ----------
models = [("Qwen2.5-0.5B-Instruct", "Qwen 0.5B"), ("Qwen2.5-1.5B-Instruct", "Qwen 1.5B"),
          ("Qwen2.5-3B-Instruct", "Qwen 3B"), ("SmolLM2-1.7B-Instruct", "SmolLM2 1.7B"),
          (args.big, "Qwen 7B (4-bit)")]
rows = []
for folder, label in models:
    path = os.path.join(args.results, folder, "baseline_raw.csv")
    if os.path.exists(path):
        raw = pd.read_csv(path)
        d = diagnostics(raw[raw.template.isin([0, 1, 2])])
        rows.append((label, d["naive_inaction_order_A_only"], d["naive_inaction_order_B_only"],
                     d["order_consistency"]))
if rows:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[1] for r in rows], 0.38, color=BLUE, label="'stay on course' shown as Case 1")
    ax.bar(x + 0.2, [r[2] for r in rows], 0.38, color=ORANGE, label="'stay on course' shown as Case 2")
    for i, r in enumerate(rows):
        ax.text(i, 0.68, f"{r[3]:.0%}", ha="center", weight="bold")
    ax.text(-0.55, 0.78, "order consistency (decision unchanged when the cases are swapped):",
            fontsize=9, color="#555")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylim(-0.8, 0.85)
    ax.set_ylabel("measured preference for inaction")
    ax.set_title("Small models answer the position, not the dilemma")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_position_bias.png"), dpi=150)
    plt.close(fig)

big = os.path.join(args.results, args.big)
# ---- fig 2: AMCE per wording / instrument -------------------------------------------
bt_path = os.path.join(big, "amce_by_template.csv")
if os.path.exists(bt_path):
    bt = pd.read_csv(bt_path)
    names = {0: "Case 1/2\n(Moral Machine)", 1: "Case 1/2\n('you decide')", 2: "Option 1/2",
             3: "Which group\nsurvives? (1/2)", 4: "Deaths only\n(1/2)", 5: "Name the\ngroup", 6: "Name,\ncalibrated"}
    temps = sorted(bt.template.unique())
    M = bt.pivot(index="dimension", columns="template", values="amce").reindex(DIMENSIONS)[temps]
    sig = bt.assign(s=(bt.ci_low > 0) | (bt.ci_high < 0)).pivot(
        index="dimension", columns="template", values="s").reindex(DIMENSIONS)[temps]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.values[i, j]
            ax.text(j, i, f"{v:+.2f}{'*' if sig.values[i, j] else ''}", ha="center", va="center",
                    fontsize=9, color="white" if abs(v) > 0.55 else "black")
    ax.set_xticks(range(len(temps)))
    ax.set_xticklabels([names[t] for t in temps], fontsize=8)
    ax.set_yticks(range(len(DIMENSIONS)))
    ax.set_yticklabels([DIMENSION_LABELS[d] for d in DIMENSIONS])
    ax.xaxis.tick_top()
    fig.colorbar(im, ax=ax, shrink=0.8, label="AMCE (* = 95% CI excludes 0)")
    ax.set_title("Qwen2.5-7B: same dilemmas, different ways of asking", pad=60)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig2_instruments.png"), dpi=150)
    plt.close(fig)

# ---- fig 3: comprehension -------------------------------------------------------------
c_path = os.path.join(big, "comprehension.csv")
if os.path.exists(c_path):
    c = pd.read_csv(c_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = {0: "Case 1/2\n(MM)", 1: "Case 1/2\n('you decide')", 2: "Option 1/2",
              3: "Which group\nsurvives?*", 4: "Deaths only"}
    ax.bar(range(len(c)), c.accuracy, color=[GREY if t == 3 else BLUE for t in c.template])
    for i, v in enumerate(c.accuracy):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center")
    ax.axhline(0.5, color=RED, ls="--", lw=1)
    ax.text(len(c) - 0.5, 0.52, "guessing", color=RED, ha="right", fontsize=9)
    ax.set_xticks(range(len(c)))
    ax.set_xticklabels([labels.get(t, t) for t in c.template], fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("knows who dies (accuracy)")
    ax.set_title("Qwen2.5-7B comprehension check\n(*ambiguous check question in that run)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig3_comprehension.png"), dpi=150)
    plt.close(fig)

# ---- fig 4: steering vs random controls ----------------------------------------------
s_path = os.path.join(big, "steering_consistency.csv")
if os.path.exists(s_path):
    cons = pd.read_csv(s_path)
    cons["same"] = cons.verdict != "inconsistent"
    cons["family"] = cons.method.map({"direction": "emotion", "neurons": "emotion",
                                      "random": "random", "random_neurons": "random"})
    g = cons.groupby(["family", "alpha"]).same.mean().mul(9)
    d = pd.read_csv(os.path.join(big, "steering_deltas.csv"))
    d["family"] = d.method.map({"direction": "emotion", "neurons": "emotion",
                                "random": "random", "random_neurons": "random"})
    a = d.assign(absd=d.delta.abs()).groupby(["family", "alpha"]).absd.mean()
    alphas = sorted(cons.alpha.unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(alphas))
    for k, (ax, series, ylab) in enumerate(zip(
            axes, (g, a), ("dimensions moving the same way\non all wordings (of 9)",
                           "mean |change of AMCE|"))):
        ax.bar(x - 0.2, [series[("emotion", al)] for al in alphas], 0.38, color=BLUE,
               label="emotion (direction + neurons)")
        ax.bar(x + 0.2, [series[("random", al)] for al in alphas], 0.38, color=GREY,
               label="random controls, same size")
        if k == 0:
            n_t = cons.n_templates.iloc[0]
            chance = 9 * 2 * 0.5 ** n_t
            ax.axhline(chance, color=RED, ls="--", lw=1)
            ax.text(len(alphas) - 0.55, chance + 0.08, "chance", color=RED, ha="right", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"alpha = {al:g}" for al in alphas])
        ax.set_ylabel(ylab)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Steering fear / anger / hope moves moral choices no more than random pushes")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig4_steering_controls.png"), dpi=150)
    plt.close(fig)
print("figures written to", args.out)

# ---- fig 5: moral map, with Qwen 7B measured by each instrument ------------------------
human_path = "data/human/CountriesChangePr.csv"
if os.path.exists(human_path):
    from moralcompass.amce import amce_table, amce_vector, scenario_effects  # noqa: E402
    from moralcompass.human_data import human_reference, load_cluster_map, load_human_amce  # noqa: E402
    from moralcompass.moral_map import distances, fit_map, project  # noqa: E402
    from moralcompass.scenarios import generate_scenarios  # noqa: E402

    human, _ = load_human_amce(human_path)
    cl_path = "data/human/country_cluster_map.csv"
    clusters = (load_cluster_map(cl_path).reindex(human.index).fillna("other")
                if os.path.exists(cl_path) else pd.Series("countries", index=human.index))
    scen = generate_scenarios(50, seed=0)

    def vec(raw, templates):
        r = raw[raw.template.isin(templates)]
        return amce_vector(amce_table(scenario_effects(r, scen), n_boot=50))

    points = {}
    for folder, label in models[:-1]:
        path = os.path.join(args.results, folder, "baseline_raw.csv")
        if os.path.exists(path):
            points[label] = vec(pd.read_csv(path), [0, 1, 2])
    raw7 = pd.read_csv(os.path.join(big, "baseline_raw.csv"))
    inst = {0: "Case 1/2 (MM)", 1: "Case 1/2 ('you decide')", 2: "Option 1/2",
            4: "Deaths only", 5: "Name the group"}
    p7 = {f"Qwen 7B · {inst[t]}": vec(raw7, [t]) for t in inst if t in raw7.template.unique()}

    m = fit_map(human)
    xy = project(m, human[DIMENSIONS].values)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    pal = {"Western": "#7fa7d9", "Eastern": "#e3a87a", "Southern": "#8cc9a5"}
    for c in sorted(clusters.unique(), key=str):
        mk = clusters.values == c
        ax.scatter(xy[mk, 0], xy[mk, 1], s=18, alpha=0.75, color=pal.get(c, "#bbb"),
                   edgecolor="none", label=f"{c} countries (humans)")
    g = project(m, human_reference(human))[0]
    ax.scatter(*g, marker="D", s=70, color="black", label="human average", zorder=4)
    marks = "o^sX"
    for i, (k, v) in enumerate(points.items()):
        ax.scatter(*project(m, v)[0], marker=marks[i % 4], s=110, color=GREY, edgecolor="black",
                   label=k, zorder=5)
    q = np.array([project(m, v)[0] for v in p7.values()])
    ax.scatter(q[:, 0], q[:, 1], s=120, color=RED, edgecolor="black", zorder=6,
               label="Qwen 7B, one point per instrument")
    for (k, _), (x0, y0) in zip(p7.items(), q):
        ax.annotate(k.split("· ")[1], (x0, y0), fontsize=8, xytext=(6, 4), textcoords="offset points")
    ev = m["pca"].explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} of the variance between countries)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
    ax.set_title("No model looks like any country — and the 7B moves with the instrument")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig5_moral_map.png"), dpi=150)
    plt.close(fig)

    ref = human_reference(human)
    dc = distances(human, ref)
    rows = [{"who": "countries: median", "distance_to_human_average": dc.median()},
            {"who": "countries: 95th percentile", "distance_to_human_average": dc.quantile(0.95)},
            {"who": f"countries: most atypical ({dc.index[-1]})", "distance_to_human_average": dc.max()}]
    for k, v in {**points, **p7}.items():
        rows.append({"who": k, "distance_to_human_average": float(np.sqrt(((v - ref) ** 2).sum()))})
    pd.DataFrame(rows).round(3).to_csv(os.path.join(args.results, "map_distances_by_instrument.csv"),
                                       index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
