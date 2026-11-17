import numpy as np

from moralcompass.amce import amce_by_template, robustness
from moralcompass.measure import run_dilemmas
from moralcompass.models import MockModel
from moralcompass.prompts import cloze_options, scenario_prompt
from moralcompass.scenarios import generate_scenarios, group_text


def test_first_listed_is_spared_if_staying():
    for s in generate_scenarios(5, seed=2):
        first_a, _ = cloze_options(s, "A")
        first_b, _ = cloze_options(s, "B")
        assert first_a == group_text(s, "other", False)   # staying kills 'ahead', spares 'other'
        assert first_b == group_text(s, "ahead", False)
        text = scenario_prompt(s, 5, "A")
        assert "Case" not in text and "Answer 1" not in text   # no numbered options


def test_cloze_runs_and_mixes_with_numbered_templates():
    scen = generate_scenarios(6, seed=0)
    raw = run_dilemmas(MockModel(), scen, templates=(0, 5))
    assert set(raw.template) == {0, 5} and len(raw) == len(scen) * 2 * 2
    c = raw[raw.template == 5]
    assert c.p_case1.between(0, 1).all() and (c.mass < 1).all()
    bt = amce_by_template(raw, scen, n_boot=100)
    rob = robustness(bt, [0, 5])
    assert set(rob.verdict) <= {"robust +", "robust -", "depends on wording"}
    # the mock spares the higher-value group: humans over pets on the cloze instrument
    sp = bt[(bt.template == 5) & (bt.dimension == "species")].amce.iloc[0]
    assert sp > 0.5


def test_calibrated_cloze_runs():
    scen = generate_scenarios(6, seed=0)
    raw = run_dilemmas(MockModel(), scen, templates=(5, 6))
    assert set(raw.template) == {5, 6} and raw.p_case1.between(0, 1).all()
    bt = amce_by_template(raw, scen, n_boot=100)
    # mock prior only likes short names; the value signal survives calibration
    assert bt[(bt.template == 6) & (bt.dimension == "species")].amce.iloc[0] > 0.5
