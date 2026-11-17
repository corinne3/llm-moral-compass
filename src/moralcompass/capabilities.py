"""Side effects of steering: does the model stay smart and fluent?

Two cheap checks, run with and without steering:
- accuracy on 30 easy multiple-choice questions (general knowledge, math),
- perplexity on neutral text (how 'surprised' the model is by normal English).
A steering effect that only appears when the model is broken is not an
effect on its morals: it is damage.
"""

from __future__ import annotations

import math

import numpy as np

QUESTIONS = [
    {"question": "What is the capital of France?", "choices": ["Berlin", "Madrid", "Paris", "Rome"], "answer": 2},
    {"question": "How much is 7 times 8?", "choices": ["54", "56", "64", "48"], "answer": 1},
    {"question": "Which planet is known as the Red Planet?", "choices": ["Mars", "Venus", "Jupiter", "Saturn"], "answer": 0},
    {"question": "What is the chemical symbol for water?", "choices": ["O2", "CO2", "H2O", "NaCl"], "answer": 2},
    {"question": "How many days are there in a leap year?", "choices": ["365", "364", "360", "366"], "answer": 3},
    {"question": "Which animal is a mammal?", "choices": ["Shark", "Dolphin", "Trout", "Octopus"], "answer": 1},
    {"question": "What is 15 plus 27?", "choices": ["42", "32", "41", "52"], "answer": 0},
    {"question": "Which gas do plants absorb from the air?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Helium"], "answer": 2},
    {"question": "What is the largest ocean on Earth?", "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "answer": 3},
    {"question": "How many sides does a hexagon have?", "choices": ["5", "6", "7", "8"], "answer": 1},
    {"question": "Who wrote 'Romeo and Juliet'?", "choices": ["Shakespeare", "Dickens", "Hugo", "Tolstoy"], "answer": 0},
    {"question": "What is the boiling point of water at sea level in Celsius?", "choices": ["90", "100", "110", "120"], "answer": 1},
    {"question": "Which is the smallest prime number?", "choices": ["0", "1", "2", "3"], "answer": 2},
    {"question": "What is the capital of Japan?", "choices": ["Seoul", "Beijing", "Bangkok", "Tokyo"], "answer": 3},
    {"question": "How much is 100 divided by 4?", "choices": ["20", "25", "40", "50"], "answer": 1},
    {"question": "Which organ pumps blood through the body?", "choices": ["Heart", "Liver", "Lung", "Kidney"], "answer": 0},
    {"question": "What is the freezing point of water in Celsius?", "choices": ["-10", "10", "0", "32"], "answer": 2},
    {"question": "Which continent is Egypt in?", "choices": ["Asia", "Europe", "South America", "Africa"], "answer": 3},
    {"question": "How many minutes are in an hour?", "choices": ["60", "100", "30", "120"], "answer": 0},
    {"question": "What color do you get by mixing blue and yellow?", "choices": ["Purple", "Green", "Orange", "Red"], "answer": 1},
    {"question": "What is the square root of 81?", "choices": ["7", "8", "9", "10"], "answer": 2},
    {"question": "Which instrument has 88 keys?", "choices": ["Guitar", "Violin", "Flute", "Piano"], "answer": 3},
    {"question": "Which is the closest star to Earth?", "choices": ["The Sun", "Sirius", "Polaris", "Vega"], "answer": 0},
    {"question": "How many legs does a spider have?", "choices": ["6", "8", "10", "12"], "answer": 1},
    {"question": "What is 9 minus 4?", "choices": ["3", "4", "5", "6"], "answer": 2},
    {"question": "Which language is mainly spoken in Brazil?", "choices": ["Spanish", "French", "English", "Portuguese"], "answer": 3},
    {"question": "What do bees make?", "choices": ["Honey", "Milk", "Silk", "Wax paper"], "answer": 0},
    {"question": "Which metal is liquid at room temperature?", "choices": ["Iron", "Mercury", "Copper", "Gold"], "answer": 1},
    {"question": "How many continents are there?", "choices": ["5", "6", "7", "8"], "answer": 2},
    {"question": "What is the opposite of 'ancient'?", "choices": ["Old", "Antique", "Historic", "Modern"], "answer": 3},
]

NEUTRAL_TEXTS = [
    "The library opens at nine in the morning and closes at six in the evening. Visitors can borrow up to five books at a time.",
    "To make a simple tomato sauce, cook chopped onions in olive oil, add crushed tomatoes and a pinch of salt, and simmer for twenty minutes.",
    "The river flows from the mountains to the sea, passing through several small towns and a large forest.",
    "A computer stores information as binary numbers, using only the digits zero and one.",
    "In spring, the farmers prepare the soil and plant seeds for the summer harvest.",
    "The museum's new exhibition presents paintings from the nineteenth century, along with letters written by the artists.",
]


def _mcq_prompt(q: dict) -> str:
    opts = "\n".join(f"{'ABCD'[i]}) {c}" for i, c in enumerate(q["choices"]))
    return f"{q['question']}\n{opts}\nAnswer with a single letter: A, B, C or D."


def capability_check(model) -> dict:
    convs = [[{"role": "user", "content": _mcq_prompt(q)}] for q in QUESTIONS]
    p = model.option_probs(convs, options=("A", "B", "C", "D"))
    answers = np.array([q["answer"] for q in QUESTIONS])
    acc = float((p[:, :4].argmax(1) == answers).mean())
    p_correct = float(p[np.arange(len(answers)), answers].mean())
    nll = model.nll(NEUTRAL_TEXTS)
    return {"accuracy": acc, "p_correct": p_correct, "mcq_mass": float(p[:, -1].mean()),
            "perplexity": float(math.exp(nll))}
