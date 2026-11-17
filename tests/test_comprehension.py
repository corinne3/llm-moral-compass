import re

from moralcompass.comprehension import comprehension_items, run_comprehension
from moralcompass.models import MockModel
from moralcompass.scenarios import generate_scenarios


def test_right_answer_is_really_the_group_that_dies():
    items = comprehension_items(generate_scenarios(5, seed=0), seed=0)
    assert len(items) == 30 * 5 * 2
    for it in items:
        text = it["messages"][0]["content"]
        k, t = it["asked_option"], it["template"]
        answer = re.search(rf"Reply {it['correct']} for: (.*)\.\n", text).group(1)
        if t == 3:   # "Answer k. <group that survives>": the right answer is the OTHER group
            survivor = re.search(rf"\nAnswer {k}\. (.*)\n", text).group(1)
            assert answer != survivor
            assert f"kill {answer}" in text
        else:
            word = "Option" if t == 2 else "Case"
            case_text = re.search(rf"{word} {k}\. (.*?)(\n|$)", text).group(1)
            assert f"death of {answer}" in case_text


def test_runs_on_mock():
    items = comprehension_items(generate_scenarios(2, seed=1), templates=(0, 3), seed=1)
    df = run_comprehension(MockModel(), items)
    assert set(df.template) == {0, 3} and df["mass"].between(0, 1).all()


def test_cloze_template_is_skipped():
    items = comprehension_items(generate_scenarios(2, seed=0), templates=(4, 5), seed=0)
    assert {it["template"] for it in items} == {4}
