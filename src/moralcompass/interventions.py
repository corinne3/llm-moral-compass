"""Plain descriptions of an activation intervention.

They only say WHAT to add and WHERE. The model backends (models.py)
know HOW to add it (PyTorch hooks for a real model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AddToResidual:
    """Add `vector` to the residual stream at the output of decoder `layer`,
    at every token position. Used for direction steering and random controls."""

    layer: int
    vector: np.ndarray  # shape (d_model,), already scaled


@dataclass
class AddToNeurons:
    """Add `values` to selected MLP neurons of decoder `layer`
    (the input of the MLP down-projection), at every token position."""

    layer: int
    indices: np.ndarray  # shape (k,), neuron indices
    values: np.ndarray   # shape (k,), amount added to each neuron
