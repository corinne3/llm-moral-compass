"""Step 0 - Check that the model loads and answers in the expected format.

Runs 6 dilemmas (one per character dimension), prints one full prompt,
the probabilities, the format validity ("mass") and the time per prompt,
then estimates how long the full baseline will take.
"""
import time

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model
from moralcompass.measure import run_dilemmas
from moralcompass.prompts import build_messages
from moralcompass.scenarios import generate_scenarios

args = base_parser(__doc__).parse_args()
model = get_model(args)
if hasattr(model, "n_layers"):
    print(f"layers={model.n_layers}  d_model={model.d_model}  mlp_neurons={model.d_mlp}")

scen = generate_scenarios(n_per_dimension=1, seed=args.seed)
example = build_messages(scen[0], template=0, order="A")
print("\n--- Example prompt (as the model sees it) ---")
print(model.format_chat(example))

t0 = time.perf_counter()
raw = run_dilemmas(model, scen, templates=(0,))
sec = (time.perf_counter() - t0) / len(raw)
print("\n--- Results ---")
print(raw[["scenario_id", "dimension", "order", "p_case1", "p_stay", "mass"]].round(3).to_string(index=False))
print(f"\nmean mass = {raw['mass'].mean():.3f}  (should be > 0.5: the model answers 1 or 2)")
print(f"time per prompt = {sec:.2f} s")
n_full = args.n_per_dim * 6 * 3 * 2
print(f"Full baseline ({args.n_per_dim}/dimension x 3 templates x 2 orders = {n_full} prompts) "
      f"~ {n_full * sec / 60:.0f} min")
