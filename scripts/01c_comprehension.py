"""Step 1c - Measurement System Analysis: does the model understand who dies?

For each wording (template), the dilemma is shown without its moral
question, followed by a factual question ("if Case k is chosen, who dies?").
Outputs in results/<model>/: comprehension_raw.csv, comprehension.csv
Read it before interpreting any moral preference measured with a wording.
"""
import os

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model, out_dir
from moralcompass.comprehension import comprehension_items, comprehension_summary, run_comprehension
from moralcompass.energy import meter
from moralcompass.scenarios import generate_scenarios

p = base_parser(__doc__)
p.set_defaults(n_per_dim=20)
p.add_argument("--templates", default="0,1,2,3,4")
args = p.parse_args()
out = out_dir(args)
templates = tuple(int(t) for t in args.templates.split(","))

model = get_model(args)
items = comprehension_items(generate_scenarios(args.n_per_dim, seed=args.seed), templates,
                            seed=args.seed)
print(f"Running {len(items)} comprehension questions ...", flush=True)
with meter("comprehension", args.model, len(items), os.path.join(out, "energy.csv"), args.cpu_watts):
    df = run_comprehension(model, items)
df.to_csv(os.path.join(out, "comprehension_raw.csv"), index=False)
summary = comprehension_summary(df)
summary.to_csv(os.path.join(out, "comprehension.csv"))
print("\nComprehension per wording (accuracy 1.0 = always knows who dies, 0.5 = guessing):")
print(summary.to_string())
print("\nPer dimension (accuracy):")
print(df.pivot_table(index="dimension", columns="template", values="is_correct").round(2).to_string())
print(f"\nSaved in {out}")
