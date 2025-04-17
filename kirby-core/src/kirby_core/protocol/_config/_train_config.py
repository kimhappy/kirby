from typing import Protocol

class _TrainConfig(Protocol):
    max_epochs: int
    vali_cycle: int
    early_stop: int
