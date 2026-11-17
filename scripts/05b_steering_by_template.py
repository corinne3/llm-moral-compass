"""Step 5b - Re-analyse a finished steering run, wording by wording (no model needed).

Reads results/<model>/steering_raw.csv and steering_meta.json, then
writes steering_deltas_by_template.csv and steering_consistency.csv.
"""
import json
import os

import pandas as pd

import _path  # noqa: F401
from moralcompass.cli import base_parser, out_dir
from moralcompass.scenarios import generate_scenarios
from moralcompass.steering import consistency, consistency_summary, deltas_by_template

args = base_parser(__doc__).parse_args()
out = out_dir(args)
raw = pd.read_csv(os.path.join(out, "steering_raw.csv"))
meta_path = os.path.join(out, "steering_meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {
    "n_per_dim": args.n_per_dim, "seed": args.seed}
scen = generate_scenarios(meta["n_per_dim"], seed=meta["seed"])
dbt = deltas_by_template(raw, scen)
dbt.to_csv(os.path.join(out, "steering_deltas_by_template.csv"), index=False)
cons = consistency(dbt)
cons.to_csv(os.path.join(out, "steering_consistency.csv"), index=False)
print(consistency_summary(cons).to_string())
real = cons[cons.method.isin(["direction", "neurons"]) & cons.verdict.ne("inconsistent")]
print("\nConsistent effects of the emotion methods:")
print(real.round(3).to_string(index=False) if len(real) else "none")
