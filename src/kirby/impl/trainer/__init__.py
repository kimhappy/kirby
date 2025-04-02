from typing import Type

from ...protocol.impl.trainer_base import TrainerBase
from .  rnn                        import RNN

_TRAINERS = [
    RNN
]

def register_trainer(trainer: Type[TrainerBase]) -> None:
    _TRAINERS.append(trainer)

def _find_trainer(name: str) -> Type[TrainerBase]:
    for trainer in _TRAINERS:
        if trainer.__name__ == name:
            return trainer

    raise ValueError(f'{ name } not found')

__all__ = [
    'register_trainer',
    '_find_trainer'
]
