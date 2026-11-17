import numpy as np

from moralcompass.amce import amce_table, scenario_effects
from moralcompass.emotions import build_dataset
from moralcompass.measure import diagnostics, run_dilemmas
from moralcompass.models import MockModel
from moralcompass.probes import choose_layer, mean_diff_directions, probe_scores, select_neurons
from moralcompass.scenarios import generate_scenarios
from moralcompass.steering import build_intervention, dose_response


def _steering_data(model):
    data = build_dataset()
    labels = np.array([d["label"] for d in data])
    acts = model.hidden_states([d["text"] for d in data])
    scores = probe_scores(acts["resid"], labels, np.array([d["situation"] for d in data]))
    layer = choose_layer(scores, model.n_layers)
    dirs = mean_diff_directions(acts["resid"], labels)
    neurons = {l: {e: dict(zip(["indices", "cohen_d", "std"], select_neurons(acts["mlp"][l], labels, e, 8)))
                   for e in ["fear", "anger", "hope"]} for l in range(model.n_layers)}
    return {"layer": layer, "resid_norm": acts["norm"],
            "directions": {l: {e: v for e, (v, _) in dd.items()} for l, dd in dirs.items()},
            "neurons": neurons,
            "neuron_std": {l: acts["mlp"][l].std(0) + 1e-8 for l in range(model.n_layers)}}, scores


def test_baseline_runs():
    m = MockModel()
    scen = generate_scenarios(5, seed=0)
    raw = run_dilemmas(m, scen)
    assert len(raw) == len(scen) * 3 * 2
    assert set(amce_table(scenario_effects(raw, scen)).dimension) >= {"species", "intervention"}
    assert 0 < diagnostics(raw)["mean_mass"] <= 1


def test_probes_find_emotions():
    m = MockModel()
    _, scores = _steering_data(m)
    assert scores[scores.target == "fear"].accuracy.max() > 0.9


def test_matched_perturbation_size():
    m = MockModel()
    sd, _ = _steering_data(m)
    layer, target = sd["layer"], 0.3 * sd["resid_norm"][sd["layer"]]
    for method in ["direction", "random"]:
        iv = build_intervention(m, method, "fear", layer, 0.3, sd)
        assert np.isclose(np.linalg.norm(iv.vector), target)
    for method in ["neurons", "random_neurons"]:
        iv = build_intervention(m, method, "fear", layer, 0.3, sd)
        footprint = m.neuron_residual_effect(layer, iv.indices, iv.values)
        assert np.isclose(np.linalg.norm(footprint), target)


def test_steering_is_removed_after_block():
    m = MockModel()
    sd, _ = _steering_data(m)
    iv = build_intervention(m, "direction", "fear", sd["layer"], 0.5, sd)
    with m.steer(iv):
        assert np.linalg.norm(m._added) > 0
    assert np.linalg.norm(m._added) == 0


def test_dose_response_shapes():
    m = MockModel()
    sd, _ = _steering_data(m)
    scen = generate_scenarios(4, seed=0)
    raw, deltas = dose_response(m, scen, sd, sd["layer"], emotions=("fear",),
                                methods=("direction", "random"), alphas=(0, 0.5))
    assert set(deltas.method) == {"direction", "random"} and len(deltas) == 2 * 9


def test_consistency_over_templates():
    from moralcompass.steering import consistency, consistency_summary, deltas_by_template
    m = MockModel()
    sd, _ = _steering_data(m)
    scen = generate_scenarios(4, seed=0)
    raw, _ = dose_response(m, scen, sd, sd["layer"], emotions=("fear",),
                           methods=("direction", "random"), alphas=(0, 0.5), templates=(0, 4))
    cons = consistency(deltas_by_template(raw, scen, n_boot=100))
    assert len(cons) == 2 * 9
    assert set(cons.verdict) <= {"sig. on all wordings", "same sign", "inconsistent"}
    assert consistency_summary(cons).shape[0] == 2


def test_neutral_frames_have_the_same_length():
    from moralcompass.emotions import FRAMES
    for i, neutral in enumerate(FRAMES["neutral"]):
        for emo in ("fear", "anger", "hope"):
            assert abs(len(neutral.split()) - len(FRAMES[emo][i].split())) <= 2, (emo, i)
