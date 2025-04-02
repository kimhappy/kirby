from typing import Protocol, Optional

class _TrainConfig(Protocol):
    max_epochs: int
    vali_cycle: int
    early_stop: Optional[int]
