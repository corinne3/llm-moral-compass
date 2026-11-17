"""Step 1 - Baseline moral preferences of one model.

Outputs in results/<model>/:
  scenarios.csv            the design (attributes of each dilemma)
  baseline_raw.csv         every probability (dilemma x template x order)
  amce.csv                 the 9 AMCEs with 95% bootstrap CIs
  diagnostics.json         position bias, order consistency, format validity
  template_sensitivity.csv AMCE per wording, and their spread
  amce.png                 bar chart (with the human average if available)
"""
import os

import _path  # noqa: F401
from moralcompass.amce import amce_by_template, amce_table, robustness, scenario_effects, template_sensitivity
from moralcompass.cli import base_parser, get_model, out_dir, save_json
from moralcompass.energy import meter
from moralcompass.human_data import human_reference, load_human_amce
from moralcompass.measure import diagnostics, diagnostics_by_template, run_dilemmas, scenario_table
from moralcompass.plots import plot_amce
from moralcompass.scenarios import generate_scenarios

p = base_parser(__doc__)
p.add_argument("--templates", default="0,1,2,3,4,5,6",
               help="0-4: numbered answers; 5: answer by name (cloze); 6: cloze calibrated")
p.add_argument("--robust-on", default=None,
               help="wordings used for the robustness verdict (default: all except 3)")
p.add_argument("--human", default="data/human/CountriesChangePr.csv")
args = p.parse_args()
out = out_dir(args)
templates = tuple(int(t) for t in args.templates.split(","))

model = get_model(args)
scen = generate_scenarios(args.n_per_dim, seed=args.seed)
scenario_table(scen).to_csv(os.path.join(out, "scenarios.csv"), index=False)

n = len(scen) * len(templates) * 2
print(f"Running {n} prompts ...", flush=True)
with meter("baseline", args.model, n, os.path.join(out, "energy.csv"), args.cpu_watts):
    raw = run_dilemmas(model, scen, templates=templates)
raw.to_csv(os.path.join(out, "baseline_raw.csv"), index=False)

tab = amce_table(scenario_effects(raw, scen))
tab.to_csv(os.path.join(out, "amce.csv"), index=False)
diag = diagnostics(raw)
save_json(diag, os.path.join(out, "diagnostics.json"))
sens = template_sensitivity(raw, scen)
sens.to_csv(os.path.join(out, "template_sensitivity.csv"))
tdiag = diagnostics_by_template(raw)
tdiag.to_csv(os.path.join(out, "template_diagnostics.csv"))
bt = amce_by_template(raw, scen)
bt.to_csv(os.path.join(out, "amce_by_template.csv"), index=False)
robust_on = ([int(x) for x in args.robust_on.split(",")] if args.robust_on
             else [t for t in templates if t != 3])
rob = robustness(bt, robust_on)
rob.to_csv(os.path.join(out, "robustness.csv"), index=False)

human = None
if os.path.exists(args.human):
    human = human_reference(load_human_amce(args.human)[0])
plot_amce({model.name: tab}, os.path.join(out, "amce.png"), human=human)

print("\nAMCE (95% CI):")
print(tab.round(3).to_string(index=False))
print("\nDiagnostics:", {k: round(v, 3) if isinstance(v, float) else v for k, v in diag.items()})
print("\nTemplate sensitivity (sd of AMCE across wordings):")
print(sens["spread_sd"].round(3).to_string())
cell = bt.assign(txt=lambda d: d.apply(
    lambda r: f"{r.amce:+.2f} [{r.ci_low:+.2f},{r.ci_high:+.2f}]", axis=1))
print("\nAMCE per wording / instrument (5 = answer by name, 6 = by name, calibrated):")
print(cell.pivot(index="dimension", columns="template", values="txt")
      .reindex(rob.dimension).to_string())
print(f"\nRobustness over wordings {robust_on}:")
print(rob.round(3).to_string(index=False))
print("\nQuality per wording (template):")
print(tdiag.to_string())
print(f"\nSaved in {out}")
