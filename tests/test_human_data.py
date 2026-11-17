import numpy as np
import pandas as pd

from moralcompass import DIMENSIONS
from moralcompass.human_data import load_human_amce

NAMES = ["[Omission -> Commission]", "[Passengers -> Pedestrians]", "Law [Illegal -> Legal]",
         "Gender [Male -> Female]", "Fitness [Large -> Fit]", "Social Status [Low -> High]",
         "Age [Elderly -> Young]", "No. Characters [Less -> More]", "Species [Pets -> Humans]"]


def test_wide_format_with_sign_flip(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for c in ["FRA", "JPN", "USA"]:
        r = {"Country": c}
        for n in NAMES:
            r[f"{n}: Estimates"] = 0.3 + rng.normal(0, 0.01)
            r[f"{n}: se"] = 0.01
        r["[Omission -> Commission]: Estimates"] *= -1      # stored as the opposite preference
        rows.append(r)
    path = tmp_path / "h.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    table, info = load_human_amce(str(path))
    assert list(table.columns) == DIMENSIONS and len(table) == 3
    assert info["flipped"] == ["intervention"] and (table.values > 0).all()


def test_tidy_format(tmp_path):
    rows = [{"country": "X", "dimension": d, "amce": 0.1} for d in DIMENSIONS]
    path = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    table, _ = load_human_amce(str(path))
    assert table.shape == (1, 9)
