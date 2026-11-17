from collections import Counter

from moralcompass import CHARACTER_DIMENSIONS
from moralcompass.prompts import scenario_prompt
from moralcompass.scenarios import describe_group, generate_scenarios


def test_balanced_design():
    scen = generate_scenarios(n_per_dimension=40, seed=1)
    assert Counter(s.dimension for s in scen) == {d: 40 for d in CHARACTER_DIMENSIONS}
    lanes = Counter(s.pos_lane for s in scen)
    assert 0.35 < lanes["ahead"] / len(scen) < 0.65          # lanes randomized


def test_groups_differ_only_on_tested_dimension():
    for s in generate_scenarios(n_per_dimension=30, seed=2):
        if s.dimension == "number":
            assert len(s.pos_group) > len(s.neg_group) and set(s.pos_group) == set(s.neg_group)
        else:
            assert len(s.pos_group) == len(s.neg_group)
        if s.dimension == "species":
            assert set(s.neg_group) <= {"dog", "cat"}


def test_reproducible():
    a = [s.to_dict() for s in generate_scenarios(10, seed=3)]
    b = [s.to_dict() for s in generate_scenarios(10, seed=3)]
    assert a == b


def test_orders_swap_the_cases():
    s = generate_scenarios(1, seed=4)[0]
    pa, pb = scenario_prompt(s, 0, "A"), scenario_prompt(s, 0, "B")
    case1_a = pa.split("Case 1. ")[1].split("\n")[0]
    case2_b = pb.split("Case 2. ")[1].split("\n")[0]
    assert case1_a == case2_b


def test_grammar():
    assert describe_group(["boy", "boy", "girl"]) == "2 boys and 1 girl"
    for s in generate_scenarios(10, seed=5):
        text = scenario_prompt(s)
        if len(s.pos_group) == 1:
            assert ", who is" in text
        if len(s.pos_group) > 1 and len(s.neg_group) > 1:
            assert ", who is" not in text
