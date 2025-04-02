from typing import Type

from ...protocol.impl.optimizer_base import OptimizerBase
from .  adam                         import Adam

_OPTIMIZERS = [
    Adam
]

def register_optimizer(optimizer: Type[OptimizerBase]) -> None:
    _OPTIMIZERS.append(optimizer)

def _find_optimizer(name: str) -> Type[OptimizerBase]:
    for optimizer in _OPTIMIZERS:
        if optimizer.__name__ == name:
            return optimizer

    raise ValueError(f'{ name } not found')

__all__ = [
    'register_optimizer',
    '_find_optimizer'
]
