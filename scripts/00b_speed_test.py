"""Step 0b - Why is it slow? A short speed diagnosis.

Prints the CPU, the PyTorch threads and the memory, then times:
  - one prompt alone (after a warm-up),
  - a batch of 8 prompts,
  - one prompt with fewer threads.
If the free memory is low (< 1 GB) while the model is loaded, Windows is
swapping to disk: close other applications (browser, Teams...).
"""
import os
import platform
import time

import _path  # noqa: F401
from moralcompass.cli import base_parser, get_model
from moralcompass.prompts import build_messages
from moralcompass.scenarios import generate_scenarios


def mem():
    try:
        import psutil
        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        return (f"RAM total {vm.total / 1e9:.1f} GB, free {vm.available / 1e9:.1f} GB, "
                f"used by this Python {rss / 1e9:.1f} GB")
    except ImportError:
        return "psutil not installed"


args = base_parser(__doc__).parse_args()
import torch  # noqa: E402

print(f"CPU: {platform.processor()}  | logical cores: {os.cpu_count()}")
print(f"PyTorch {torch.__version__}, threads used: {torch.get_num_threads()}")
print("Before loading:", mem())
t0 = time.perf_counter()
model = get_model(args)
print(f"Model loaded in {time.perf_counter() - t0:.0f} s")
print("After loading: ", mem())

msgs = [build_messages(s) for s in generate_scenarios(2, seed=0)][:8]
model.option_probs(msgs[:1])                       # warm-up (first call is slower)


def timed(label, convs):
    t = time.perf_counter()
    model.option_probs(convs)
    dt = time.perf_counter() - t
    print(f"{label:35s} {dt:7.2f} s  -> {dt / len(convs):6.2f} s per prompt")


timed("1 prompt", msgs[:1])
timed("batch of 8 prompts", msgs[:8])
n = torch.get_num_threads()
for k in sorted({max(1, n // 2), 4} - {n}):
    torch.set_num_threads(k)
    timed(f"1 prompt with {k} threads", msgs[:1])
torch.set_num_threads(n)
print("During the run: ", mem())
