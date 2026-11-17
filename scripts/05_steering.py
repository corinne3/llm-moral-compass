"""Step 5 - Steering: push fear / anger / hope into the model, measure morals.

For each method (direction, neurons, random, random_neurons), emotion and
alpha, the same dilemmas are measured and each AMCE change is compared to
alpha = 0 on the same dilemmas (paired bootstrap CI).

Outputs in results/<model>/: steering_raw.csv, steering_deltas.csv,
dose_<emotion>.png, steering_heatmap.png, steered_samples.txt
"""
import os

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model, load_pickle, out_dir
from moralcompass.energy import meter
from moralcompass.plots import plot_delta_heatmap, plot_dose_response
from moralcompass.scenarios import generate_scenarios
from moralcompass.cli import save_json
from moralcompass.steering import (METHODS, consistency, consistency_summary, deltas_by_template,
                                   dose_response, steered_samples)

p = base_parser(__doc__)
p.set_defaults(n_per_dim=20)
p.add_argument("--alphas", default="0,0.1,0.25,0.5")
p.add_argument("--methods", default=",".join(METHODS))
p.add_argument("--emotions", default="fear,anger,hope")
p.add_argument("--layer", type=int, default=None)
p.add_argument("--templates", default="0")
args = p.parse_args()
out = out_dir(args)

sd = load_pickle(os.path.join(out, "steering_data.pkl"))
layer = args.layer if args.layer is not None else sd["layer"]
alphas = tuple(float(a) for a in args.alphas.split(","))
methods = args.methods.split(",")
emotions = args.emotions.split(",")
templates = tuple(int(t) for t in args.templates.split(","))

model = get_model(args)
scen = generate_scenarios(args.n_per_dim, seed=args.seed)
n_cond = len(methods) * len(emotions) * len([a for a in alphas if a > 0]) + 1
n = n_cond * len(scen) * len(templates) * 2
print(f"Layer {layer}: {n_cond} conditions, {n} prompts in total", flush=True)


def progress(i, total, m, e, a):
    print(f"  [{i}/{total}] {m} / {e} / alpha={a}", flush=True)


with meter("steering", args.model, n, os.path.join(out, "energy.csv"), args.cpu_watts):
    raw, deltas = dose_response(model, scen, sd, layer, emotions=emotions, methods=methods,
                                alphas=alphas, templates=templates, seed=args.seed,
                                progress=progress)
raw.to_csv(os.path.join(out, "steering_raw.csv"), index=False)
save_json({"n_per_dim": args.n_per_dim, "seed": args.seed, "layer": layer,
           "templates": list(templates)}, os.path.join(out, "steering_meta.json"))
deltas.to_csv(os.path.join(out, "steering_deltas.csv"), index=False)
for e in emotions:
    plot_dose_response(deltas, e, os.path.join(out, f"dose_{e}.png"))
top = max(alphas)
sel = deltas[deltas.alpha == top].assign(condition=lambda d: d.method + " / " + d.emotion)
plot_delta_heatmap(sel, os.path.join(out, "steering_heatmap.png"),
                   title=f"Steering at alpha={top}: change of AMCE")

# Specificity: is the emotion effect bigger than a random push of the same size?
g = deltas.groupby(["method", "emotion", "alpha"])["delta"].apply(lambda s: s.abs().mean())
print("\nMean |Δ AMCE| over the 9 dimensions:")
print(g.unstack("alpha").round(3).to_string())
print("\nPosition bias per condition (baseline: "
      f"{deltas['base_position_bias'].iloc[0]:+.3f}):")
print(deltas.groupby(["method", "emotion", "alpha"])["position_bias"].first().unstack("alpha").round(3).to_string())
print("\nMean format validity (mass) per condition:")
print(deltas.groupby(["method", "emotion", "alpha"])["mean_mass"].first().unstack("alpha").round(3).to_string())

if len(templates) > 1:
    dbt = deltas_by_template(raw, scen)
    dbt.to_csv(os.path.join(out, "steering_deltas_by_template.csv"), index=False)
    cons = consistency(dbt)
    cons.to_csv(os.path.join(out, "steering_consistency.csv"), index=False)
    print(f"\nConsistency over wordings {list(templates)}: number of dimensions (out of 9)")
    print("sig_all = CI excludes 0 on every wording; same_sign = same direction on every wording.")
    print("The random controls show how many you get BY CHANCE.")
    print(consistency_summary(cons).to_string())
    real = cons[cons.method.isin(["direction", "neurons"]) & cons.verdict.ne("inconsistent")]
    print("\nConsistent effects of the emotion methods:")
    print(real.sort_values(["emotion", "method", "alpha"]).round(3).to_string(index=False)
          if len(real) else "none")

with open(os.path.join(out, "steered_samples.txt"), "w", encoding="utf-8") as f:
    for a in [x for x in alphas if x > 0]:
        for k, v in steered_samples(model, sd, layer, a).items():
            f.write(f"=== {k}\n{v.strip()}\n\n")
print(f"\nSaved in {out} (read steered_samples.txt: does the model *sound* emotional?)")
