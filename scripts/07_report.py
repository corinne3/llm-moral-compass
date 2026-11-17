"""Step 7 - Write results/<model>/REPORT.md from all the saved outputs."""
import json
import os

import pandas as pd

import _path  # noqa: F401
from moralcompass import DIMENSION_LABELS
from moralcompass.cli import base_parser, out_dir

args = base_parser(__doc__).parse_args()
out = out_dir(args)


def read(name):
    path = os.path.join(out, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def img(name):
    return f"![{name}]({name})\n" if os.path.exists(os.path.join(out, name)) else ""


L = [f"# Results - {args.model}\n"]
diag_path = os.path.join(out, "diagnostics.json")
raw_path = os.path.join(out, "baseline_raw.csv")
if os.path.exists(raw_path):          # recompute: older runs lack the newest diagnostics
    from moralcompass.measure import diagnostics
    from moralcompass.cli import save_json
    save_json(diagnostics(pd.read_csv(raw_path)), diag_path)
if os.path.exists(diag_path):
    d = json.load(open(diag_path))
    L += ["## 1. Measurement quality\n",
          f"- Position bias (P('1') - 0.5): **{d['position_bias']:+.3f}**",
          f"- Order consistency (same decision when cases are swapped): **{d['order_consistency']:.0%}**",
          f"- Format validity (mean probability on '1'/'2'): **{d['mean_mass']:.2f}**",
          f"- Prompts: {d['n_prompts']}"]
    if "naive_inaction_order_A_only" in d:
        L += [f"- What a single-order study would conclude about 'inaction': "
              f"**{d['naive_inaction_order_A_only']:+.2f}** (stay shown as Case 1) or "
              f"**{d['naive_inaction_order_B_only']:+.2f}** (stay shown as Case 2). "
              f"Counterbalanced value: see the AMCE table below.\n"]
amce = read("amce.csv")
if amce is not None:
    amce["preference"] = amce.dimension.map(DIMENSION_LABELS)
    L += ["## 2. Baseline moral preferences (AMCE, 95% CI)\n",
          amce[["preference", "amce", "ci_low", "ci_high", "n"]].round(3).to_markdown(index=False),
          "", img("amce.png")]
rob = read("robustness.csv")
if rob is not None:
    rob["preference"] = rob.dimension.map(DIMENSION_LABELS)
    L += ["### Which preferences survive a change of wording / instrument?\n",
          rob[["preference", "min", "max", "verdict"]].round(3).to_markdown(index=False), ""]
bt = read("amce_by_template.csv")
if bt is not None:
    piv = bt.assign(txt=bt.apply(lambda r: f"{r.amce:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
    L += ["### AMCE per wording (5 = answer by name)\n",
          piv.pivot(index="dimension", columns="template", values="txt").to_markdown(), ""]
comp = read("comprehension.csv")
if comp is not None:
    L += ["### Comprehension check (does the model know who dies?)\n",
          comp.to_markdown(index=False), ""]
sens = read("template_sensitivity.csv")
if sens is not None:
    L += ["### Sensitivity to the wording\n", sens.round(3).to_markdown(index=False), ""]
pdl = read("persuasion_deltas.csv")
if pdl is not None:
    s = pdl[pdl.significant]
    L += ["## 3. Prompting (significant changes only)\n",
          s[["condition", "dimension", "delta", "ci_low", "ci_high"]].round(3).to_markdown(index=False)
          if len(s) else "No significant change.", "", img("persuasion_heatmap.png")]
al = read("cultural_alignment.csv")
if al is not None:
    L += ["### Cultural prompting\n", al.to_markdown(index=False), ""]
ps = read("probe_scores.csv")
if ps is not None:
    L += ["## 4. Emotions inside the model\n", img("probes.png")]
sdl = read("steering_deltas.csv")
if sdl is not None:
    spec = (sdl.groupby(["method", "emotion", "alpha"])["delta"].apply(lambda x: x.abs().mean())
            .unstack("method").round(3))
    L += ["## 5. Steering\n", "Mean |Δ AMCE| over the 9 dimensions "
          "(an emotion method should beat its random control):\n",
          spec.to_markdown(), "", img("steering_heatmap.png")]
    for e in sdl.emotion.unique():
        L.append(img(f"dose_{e}.png"))
se = read("side_effects.csv")
if se is not None:
    L += ["## 6. Side effects\n", img("side_effects.png")]
en = read("energy.csv")
if en is not None:
    L += ["## 7. Time and energy\n", en.to_markdown(index=False), ""]
with open(os.path.join(out, "REPORT.md"), "w") as f:
    f.write("\n".join(L))
print(f"Written {os.path.join(out, 'REPORT.md')}")
