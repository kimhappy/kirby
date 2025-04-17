from typing import Protocol

class _GeneralConfig(Protocol):
    seed         : int
    device       : str
    deterministic: bool
    compile      : bool
    num_cond     : int
    sample_rate  : int
