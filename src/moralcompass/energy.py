"""Measure time (and, if possible, energy) of each step.

- If the 'codecarbon' package is installed, it estimates energy from the
  CPU/RAM (offline mode, country = France by default).
- Otherwise, if you give your CPU power in watts (--cpu-watts), energy is
  estimated as power x time. Rough, but enough to compare steps.
"""

from __future__ import annotations

import csv
import os
import time
from contextlib import contextmanager


@contextmanager
def meter(step: str, model_name: str, n_forward: int, log_path: str,
          cpu_watts: float | None = None, country: str = "FRA"):
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker  # optional
        tracker = OfflineEmissionsTracker(country_iso_code=country, log_level="error",
                                          save_to_file=False)
        tracker.start()
    except Exception:
        tracker = None
    t0 = time.perf_counter()
    record: dict = {"step": step, "model": model_name, "n_forward": n_forward}
    try:
        yield record
    finally:
        sec = time.perf_counter() - t0
        kwh = None
        if tracker is not None:
            try:
                tracker.stop()
                kwh = float(tracker.final_emissions_data.energy_consumed)
            except Exception:
                kwh = None
        if kwh is None and cpu_watts:
            kwh = cpu_watts * sec / 3.6e6
        record.update({"seconds": round(sec, 2),
                       "sec_per_forward": round(sec / max(record["n_forward"], 1), 4),
                       "kwh": kwh,
                       "wh_per_1000_forward": None if kwh is None
                       else round(kwh * 1000 * 1000 / max(record["n_forward"], 1), 4)})
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        new = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(record))
            if new:
                w.writeheader()
            w.writerow(record)
