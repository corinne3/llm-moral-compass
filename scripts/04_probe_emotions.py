"""Step 4 - Find fear, anger and hope inside the model.

1. Run 320 matched sentences (neutral / fear / anger / hope) and record,
   for every layer, the mean residual activation and the MLP neurons.
2. Train linear probes layer by layer (tested on unseen situations).
3. Choose the steering layer, compute emotion directions (difference of
   means) and the top-k emotion neurons.

Outputs in results/<model>/: probe_scores.csv, probes.png,
steering_data.pkl (everything step 5 needs)
"""
import os

import numpy as np
import pandas as pd

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model, out_dir, save_json, save_pickle
from moralcompass.emotions import EMOTIONS, build_dataset
from moralcompass.energy import meter
from moralcompass.plots import plot_probes
from moralcompass.probes import choose_layer, mean_diff_directions, probe_scores, select_neurons

p = base_parser(__doc__)
p.add_argument("--top-k", type=int, default=32, help="neurons per emotion")
p.add_argument("--layer", type=int, default=None, help="force the steering layer")
args = p.parse_args()
out = out_dir(args)

model = get_model(args)
data = build_dataset()
texts = [d["text"] for d in data]
labels = np.array([d["label"] for d in data])
groups = np.array([d["situation"] for d in data])

with meter("probe_activations", args.model, len(texts), os.path.join(out, "energy.csv"), args.cpu_watts):
    acts = model.hidden_states(texts, mlp=True)

scores = probe_scores(acts["resid"], labels, groups)
scores.to_csv(os.path.join(out, "probe_scores.csv"), index=False)
layer = args.layer if args.layer is not None else choose_layer(scores, model.n_layers)
plot_probes(scores, os.path.join(out, "probes.png"), chosen_layer=layer)

dirs = mean_diff_directions(acts["resid"], labels)
neurons, neuron_std = {}, {}
for l in range(model.n_layers):
    neuron_std[l] = acts["mlp"][l].std(0, ddof=1) + 1e-8
    neurons[l] = {}
    for e in EMOTIONS:
        idx, d, sd = select_neurons(acts["mlp"][l], labels, e, k=args.top_k)
        neurons[l][e] = {"indices": idx, "cohen_d": d, "std": sd}

steering_data = {
    "model": args.model, "layer": layer,
    "resid_norm": acts["norm"],
    "directions": {l: {e: v for e, (v, _) in dd.items()} for l, dd in dirs.items()},
    "direction_raw_norm": {l: {e: n for e, (_, n) in dd.items()} for l, dd in dirs.items()},
    "neurons": neurons, "neuron_std": neuron_std,
}
save_pickle(steering_data, os.path.join(out, "steering_data.pkl"))

# How similar are the three emotion directions at the chosen layer?
V = np.stack([steering_data["directions"][layer][e] for e in EMOTIONS])
cos = pd.DataFrame(V @ V.T, index=EMOTIONS, columns=EMOTIONS)
save_json({"layer": layer, "cosine_between_emotions": cos.round(3).to_dict()},
          os.path.join(out, "probe_summary.json"))

s = scores[(scores.layer == layer)]
print(f"\nChosen steering layer: {layer} / {model.n_layers}")
print(s.round(3).to_string(index=False))
print("\nCosine similarity between emotion directions at that layer:")
print(cos.round(2).to_string())
off = cos.values[~np.eye(3, dtype=bool)]
if off.max() > 0.9:
    print("\nWARNING: the emotion directions are almost identical (cosine > 0.9). They capture a"
          " shared feature (length, 'emotional vs neutral'...), not specific emotions.")
print(f"\nTypical token norm at layer {layer} (median): {acts['norm'][layer]:.1f}")
print(f"\nSaved in {out}")
