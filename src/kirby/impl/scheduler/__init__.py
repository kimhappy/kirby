from typing import Type

from ...protocol.impl.scheduler_base import SchedulerBase
from .  plateau                      import Plateau

_SCHEDULERS = [
    Plateau
]

def register_scheduler(scheduler: Type[SchedulerBase]) -> None:
    _SCHEDULERS.append(scheduler)

def _find_scheduler(name: str) -> Type[SchedulerBase]:
    for scheduler in _SCHEDULERS:
        if scheduler.__name__ == name:
            return scheduler

    raise ValueError(f'{ name } not found')

__all__ = [
    'register_scheduler',
    '_find_scheduler'
]
