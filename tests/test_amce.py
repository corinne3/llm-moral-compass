import numpy as np
import pandas as pd

from moralcompass.amce import amce_table, bootstrap_mean, paired_delta, scenario_effects
from moralcompass.scenarios import generate_scenarios


def fake_raw(scen, p_pos=0.8, position_bias=0.1):
    """A model that spares the 'pos' group with prob p_pos and also
    likes answer '1' (+position_bias). Counterbalancing must cancel it."""
    rows = []
    for s in scen:
        p_stay = p_pos if s.pos_lane == "other" else 1 - p_pos
        for order in "AB":
            p1 = (p_stay if order == "A" else 1 - p_stay) + position_bias
            p_stay_obs = p1 if order == "A" else 1 - p1
            rows.append({"scenario_id": s.id, "dimension": s.dimension, "template": 0,
                         "order": order, "p_case1": p1, "mass": 1.0, "p_stay": p_stay_obs})
    return pd.DataFrame(rows)


def test_amce_recovers_truth_despite_position_bias():
    scen = generate_scenarios(50, seed=0)
    tab = amce_table(scenario_effects(fake_raw(scen), scen)).set_index("dimension")
    for d in ["species", "number", "age", "social_status", "fitness", "gender"]:
        assert abs(tab.loc[d, "amce"] - 0.6) < 1e-9          # 2 * 0.8 - 1
    assert abs(tab.loc["intervention", "amce"]) < 0.2         # lanes randomized


def test_bootstrap_ci_contains_mean():
    m, lo, hi = bootstrap_mean(np.random.default_rng(0).normal(1, 1, 200))
    assert lo < m < hi and abs(m - 1) < 0.3


def test_paired_delta_detects_shift():
    scen = generate_scenarios(30, seed=1)
    base = scenario_effects(fake_raw(scen, 0.7), scen)
    cond = scenario_effects(fake_raw(scen, 0.8), scen)
    d = paired_delta(base, cond).set_index("dimension")
    assert abs(d.loc["age", "delta"] - 0.2) < 1e-9 and d.loc["age", "significant"]
