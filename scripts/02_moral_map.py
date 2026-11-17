"""Step 2 - Place the models on the moral map of 130 countries.

Needs the human file (see README, 'Human data'). Uses every
results/<model>/amce.csv found (or the ones given with --models).
Outputs in results/: moral_map.png, map_distances.csv,
map_loadings.csv, country_clusters.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd

import _path  # noqa: F401
from moralcompass import DIMENSIONS
from moralcompass.amce import amce_vector
from moralcompass.human_data import human_reference, load_cluster_map, load_human_amce
from moralcompass.moral_map import cluster_countries, distances, nearest_cluster, plot_map

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--results", default="results")
p.add_argument("--human", default="data/human/CountriesChangePr.csv")
p.add_argument("--models", nargs="*", help="result folder names (default: all)")
p.add_argument("--clusters", default="data/human/country_cluster_map.csv",
               help="official cultural clusters (OSF); Ward clustering is used if missing")
p.add_argument("--label-countries", nargs="*",
               default=["France", "FRA", "Japan", "JPN", "United States", "USA", "Brazil", "BRA",
                        "China", "CHN", "Germany", "DEU", "India", "IND", "Colombia", "COL"])
args = p.parse_args()

if not os.path.exists(args.human):
    sys.exit(f"Human file not found: {args.human}\nSee README section 'Human data'.")
human, info = load_human_amce(args.human)
print(f"{info['n_countries']} countries loaded. Sign flips: {info['flipped'] or 'none'}")

folders = args.models or sorted(os.path.basename(os.path.dirname(f))
                                for f in glob.glob(os.path.join(args.results, "*", "amce.csv")))
models = {f: amce_vector(pd.read_csv(os.path.join(args.results, f, "amce.csv"))) for f in folders}
if not models:
    sys.exit("No results/<model>/amce.csv found: run 01_baseline.py first.")

if os.path.exists(args.clusters):
    clusters = load_cluster_map(args.clusters).reindex(human.index).fillna("other")
    print("Using the official clusters of Awad et al. (Western / Eastern / Southern)")
else:
    clusters = cluster_countries(human)
clusters.to_csv(os.path.join(args.results, "country_clusters.csv"))
res = plot_map(human, models, os.path.join(args.results, "moral_map.png"), clusters=clusters,
               label_countries=args.label_countries,
               title="LLMs on the moral map of countries (PCA of 9 AMCEs)")
res["loadings"].to_csv(os.path.join(args.results, "map_loadings.csv"))

ref = human_reference(human)
rows = []
for name, vec in models.items():
    d = distances(human, vec)
    near = nearest_cluster(human, clusters, vec)
    rows.append({"model": name,
                 "distance_to_human_average": float(((vec - ref) ** 2).sum() ** 0.5),
                 "nearest_countries": ", ".join(d.index[:5]),
                 "nearest_cluster": min(near, key=near.get),
                 **{f"dist_cluster_{k}": round(v, 3) for k, v in near.items()}})
table = pd.DataFrame(rows)
table.to_csv(os.path.join(args.results, "map_distances.csv"), index=False)
print(table.to_string(index=False))
print("\nPCA loadings (what each axis means):")
print(res["loadings"].round(2).to_string())
print("\nCluster profiles (mean AMCE):")
print(human.groupby(clusters)[DIMENSIONS].mean().round(2).to_string())
