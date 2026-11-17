"""Step 5a - Before steering: at which strength does the model break?

For a grid of alpha values, with the 'fear' direction and a random direction
of the same size, measures:
  - mass: does the model still answer "1"/"2" (template 1, 12 dilemmas)?
  - accuracy on 30 easy questions,
  - a short sentence written by the steered model.
Recommends the alphas to use in step 5: the model must still work,
otherwise any 'moral' change is just damage.
Output: results/<model>/alpha_scan.csv
"""
import os

import pandas as pd

import _path  # noqa: F401
from moralcompass.capabilities import capability_check
from moralcompass.cli import base_parser, get_model, load_pickle, out_dir
from moralcompass.measure import run_dilemmas
from moralcompass.scenarios import generate_scenarios
from moralcompass.steering import build_intervention

p = base_parser(__doc__)
p.add_argument("--alphas", default="0,0.02,0.05,0.1,0.2,0.35,0.5,1.0")
p.add_argument("--emotion", default="fear")
p.add_argument("--layer", type=int, default=None)
args = p.parse_args()
out = out_dir(args)
sd = load_pickle(os.path.join(out, "steering_data.pkl"))
layer = args.layer if args.layer is not None else sd["layer"]
model = get_model(args)
scen = generate_scenarios(2, seed=args.seed)
msgs = [{"role": "user", "content": "How do you feel right now? Answer in one sentence."}]

rows = []
for a in (float(x) for x in args.alphas.split(",")):
    for method in (["direction", "random"] if a > 0 else ["none"]):
        iv = build_intervention(model, method, args.emotion, layer, a, sd) if a > 0 else []
        with model.steer(iv):
            mass = run_dilemmas(model, scen, templates=(1,))["mass"].mean()
            cap = capability_check(model)
            text = model.generate(msgs, max_new_tokens=30).strip().replace("\n", " ")
        rows.append({"alpha": a, "method": method, "mass": round(mass, 3),
                     "accuracy": cap["accuracy"], "perplexity": round(cap["perplexity"], 1),
                     "sample": text[:90]})
        print(f"alpha={a:<5} {method:9s} mass={mass:.3f} acc={cap['accuracy']:.2f} "
              f"ppl={cap['perplexity']:8.1f} | {text[:70]}", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(out, "alpha_scan.csv"), index=False)

base = df[df.alpha == 0].iloc[0]
d = df[df.method == "direction"]
safe = d[(d.mass >= 0.9 * base.mass) & (d.accuracy >= base.accuracy - 0.1)].alpha.tolist()
if safe:
    hi = max(safe)
    lo = max([a for a in safe if a <= hi / 2] or [hi / 2])
    print(f"\nSafe alphas (mass >= 90% of baseline, accuracy within 10 points): {safe}")
    print(f"Recommended for step 5:  --alphas 0,{lo:g},{hi:g}")
else:
    print("\nNo safe alpha in this grid: try smaller values, e.g. --alphas 0,0.005,0.01,0.02")
