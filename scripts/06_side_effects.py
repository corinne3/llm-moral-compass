"""Step 6 - Side effects: does steering damage the model?

For each method / emotion / alpha: accuracy on 30 easy questions and
perplexity on neutral text. Output: side_effects.csv, side_effects.png
"""
import os

import pandas as pd

import _path  # noqa: F401
from moralcompass.capabilities import capability_check
from moralcompass.cli import base_parser, get_model, load_pickle, out_dir
from moralcompass.plots import plot_side_effects
from moralcompass.steering import METHODS, build_intervention

p = base_parser(__doc__)
p.add_argument("--alphas", default="0,0.1,0.25,0.5,1.0")
p.add_argument("--methods", default=",".join(METHODS))
p.add_argument("--emotions", default="fear,anger,hope")
p.add_argument("--layer", type=int, default=None)
args = p.parse_args()
out = out_dir(args)

sd = load_pickle(os.path.join(out, "steering_data.pkl"))
layer = args.layer if args.layer is not None else sd["layer"]
model = get_model(args)
base = capability_check(model)
path = os.path.join(out, "side_effects.csv")
rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []   # resume
done = {(r["method"], r["emotion"], float(r["alpha"])) for r in rows}
for m in args.methods.split(","):
    for e in args.emotions.split(","):
        for a in (float(x) for x in args.alphas.split(",")):
            if (m, e, a) in done:
                continue
            if a == 0:
                r = base
            else:
                with model.steer(build_intervention(model, m, e, layer, a, sd, args.seed)):
                    r = capability_check(model)
            rows.append({"method": m, "emotion": e, "alpha": a, **r})
            pd.DataFrame(rows).to_csv(path, index=False)          # save after each condition
            print(f"{m:15s} {e:6s} alpha={a:<5} acc={r['accuracy']:.2f} ppl={r['perplexity']:.1f}",
                  flush=True)
df = pd.DataFrame(rows)
df.to_csv(path, index=False)
plot_side_effects(df, os.path.join(out, "side_effects.png"))
print(f"\nSaved in {out}")
