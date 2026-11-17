"""Step 3 - Can words move the model? Conversational prompt engineering.

Conditions (see moralcompass/persuasion.py): emotional conversation
(fear / anger / hope), ethical personas, cultural personas.
Every condition runs on the SAME dilemmas as the baseline, so each change
of AMCE is measured paired, with a bootstrap CI.

Outputs in results/<model>/: persuasion_raw.csv, persuasion_deltas.csv,
persuasion_amce.csv, persuasion_heatmap.png, cultural_alignment.csv
"""
import os

import pandas as pd

import _path  # noqa: F401
from moralcompass.amce import amce_table, amce_vector, paired_delta, scenario_effects
from moralcompass.cli import base_parser, get_model, out_dir
from moralcompass.energy import meter
from moralcompass.human_data import load_human_amce
from moralcompass.measure import run_dilemmas
from moralcompass.moral_map import alignment_rank
from moralcompass.persuasion import DEFAULT_COUNTRIES, conditions
from moralcompass.plots import plot_delta_heatmap
from moralcompass.scenarios import generate_scenarios

ISO3 = {"France": "FRA", "Japan": "JPN", "United States": "USA", "Brazil": "BRA",
        "China": "CHN", "Germany": "DEU", "India": "IND", "Colombia": "COL"}

p = base_parser(__doc__)
p.set_defaults(n_per_dim=30)
p.add_argument("--templates", default="0")
p.add_argument("--countries", nargs="*", default=DEFAULT_COUNTRIES)
p.add_argument("--human", default="data/human/CountriesChangePr.csv")
args = p.parse_args()
out = out_dir(args)
templates = tuple(int(t) for t in args.templates.split(","))

model = get_model(args)
scen = generate_scenarios(args.n_per_dim, seed=args.seed)
conds = conditions(args.countries)
# Resume: conditions already saved in persuasion_raw.csv are not recomputed
# (Colab can disconnect; each condition is saved as soon as it is done).
raw_path = os.path.join(out, "persuasion_raw.csv")
done = pd.read_csv(raw_path) if os.path.exists(raw_path) else pd.DataFrame()
key = {"templates": sorted(templates), "n_per_dim": args.n_per_dim, "seed": args.seed}
if len(done) and (sorted(done.template.unique().tolist()) != key["templates"]
                  or done.scenario_id.nunique() != len(scen)):
    print("Existing persuasion_raw.csv was made with other settings: starting again.")
    done = pd.DataFrame()
raws, effects = ([done] if len(done) else []), {}
for name in (done.condition.unique() if len(done) else []):
    effects[name] = scenario_effects(done[done.condition == name], scen)
for i, (name, c) in enumerate(conds.items()):
    if name in effects:
        print(f"[{i + 1}/{len(conds)}] {name}: already done, skipped", flush=True)
        continue
    n = len(scen) * len(templates) * 2
    print(f"[{i + 1}/{len(conds)}] {name} ({n} prompts)", flush=True)
    with meter(f"persuasion:{name}", args.model, n, os.path.join(out, "energy.csv"), args.cpu_watts):
        r = run_dilemmas(model, scen, templates=templates, system=c["system"],
                         context=c["context"], condition=name)
    raws.append(r)
    effects[name] = scenario_effects(r, scen)
    pd.concat(raws, ignore_index=True).to_csv(raw_path, index=False)   # save now
raw = pd.concat(raws, ignore_index=True)

amces = pd.concat([amce_table(e).assign(condition=k) for k, e in effects.items()])
amces.to_csv(os.path.join(out, "persuasion_amce.csv"), index=False)
deltas = pd.concat([paired_delta(effects["baseline"], e).assign(condition=k)
                    for k, e in effects.items() if k != "baseline"])
deltas.to_csv(os.path.join(out, "persuasion_deltas.csv"), index=False)
plot_delta_heatmap(deltas, os.path.join(out, "persuasion_heatmap.png"),
                   title="Prompting: change of AMCE vs baseline")
print(deltas[deltas.significant][["condition", "dimension", "delta", "ci_low", "ci_high"]]
      .round(3).to_string(index=False))

if os.path.exists(args.human):
    human, _ = load_human_amce(args.human)
    base_vec = amce_vector(amce_table(effects["baseline"]))
    rows = []
    for c in args.countries:
        key = c if c in human.index else ISO3.get(c)
        if key not in human.index:
            print(f"  {c}: not found in the human file, skipped")
            continue
        vec = amce_vector(amce_table(effects[f"country_{c}"]))
        before, after = alignment_rank(human, key, base_vec), alignment_rank(human, key, vec)
        rows.append({"country": c, "rank_before": before["rank"], "rank_after": after["rank"],
                     "dist_before": round(before["distance"], 3),
                     "dist_after": round(after["distance"], 3), "n_countries": after["n"]})
    al = pd.DataFrame(rows)
    al.to_csv(os.path.join(out, "cultural_alignment.csv"), index=False)
    print("\nCultural prompting (rank 1 = the model is closest to that very country):")
    print(al.to_string(index=False))
