from typing import List
import numpy as np

def _merge_cond(
    input: np.ndarray,
    conds: List[float]) -> np.ndarray:
    conds    = np.array(conds, dtype = np.float32)
    expanded = np.expand_dims(input, axis = 1)
    tiled    = np.tile(conds, (input.size, 1))
    ret      = np.concatenate((expanded, tiled), axis = 1)
    return ret
