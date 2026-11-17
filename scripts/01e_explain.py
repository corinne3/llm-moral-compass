"""Step 1e - Ask the model to explain its choice, in words (qualitative check).

This is how the 'label binding' artefact was found: the model reasoned
"protect those who obey the law", then answered with the number of the
case that killed them. Run it on any wording and dimension.
Output: results/<model>/explanations_t<template>_<dimension>.txt
"""
import os

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model, out_dir
from moralcompass.prompts import build_messages
from moralcompass.scenarios import generate_scenarios

p = base_parser(__doc__)
p.add_argument("--template", type=int, default=4)
p.add_argument("--dimension", default="law",
               help="species, number, age, social_status, fitness, gender, or law")
p.add_argument("--n", type=int, default=6)
args = p.parse_args()
out = out_dir(args)

scen = generate_scenarios(50, seed=args.seed)
if args.dimension == "law":
    scen = [s for s in scen if s.signals]
else:
    scen = [s for s in scen if s.dimension == args.dimension]
model = get_model(args)
path = os.path.join(out, f"explanations_t{args.template}_{args.dimension}.txt")
with open(path, "w", encoding="utf-8") as f:
    for s in scen[:args.n]:
        msgs = build_messages(s, args.template, "A")
        q = msgs[-1]["content"]
        if args.template == 5:
            q += "\n\nExplain your reasoning in 3 sentences, then give the group to spare, exactly as written."
        else:   # replace the final instruction ("answer 1 or 2") by a request to explain
            q = (q.rsplit("\n\n", 1)[0] + "\n\nWhich option should be chosen? Explain your "
                 "reasoning in 3 sentences, then give your answer: 1 or 2.")
        msgs[-1]["content"] = q
        ans = model.generate(msgs, max_new_tokens=180)
        block = f"{msgs[-1]['content']}\n\n-> ANSWER: {ans.strip()}\n{'=' * 90}\n"
        print(block)
        f.write(block)
print(f"Saved in {path}")
