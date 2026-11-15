"""Reproducible seeds."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    except ImportError:
        pass
