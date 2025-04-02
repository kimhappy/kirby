from typing import Protocol, List

import numpy as np

class _Data(Protocol):
    cond  : List[float]
    input : np.ndarray
    output: np.ndarray
