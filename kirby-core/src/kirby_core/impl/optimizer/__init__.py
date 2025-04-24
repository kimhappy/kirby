from typing import Type

from .  adam     import Adam
from ...protocol import OptimizerBase

_OPTIMIZERS = [
    Adam
]

def register_optimizer(optimizer: Type[OptimizerBase]) -> None:
    _OPTIMIZERS.append(optimizer)

def _find_optimizer(name: str) -> Type[OptimizerBase]:
    for optimizer in _OPTIMIZERS:
        if optimizer.__name__ == name:
            return optimizer

    raise ValueError(f'{name} not found')
