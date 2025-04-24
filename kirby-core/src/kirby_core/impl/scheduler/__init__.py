from typing import Type

from .  plateau  import Plateau
from ...protocol import SchedulerBase

_SCHEDULERS = [
    Plateau
]

def register_scheduler(scheduler: Type[SchedulerBase]) -> None:
    _SCHEDULERS.append(scheduler)

def _find_scheduler(name: str) -> Type[SchedulerBase]:
    for scheduler in _SCHEDULERS:
        if scheduler.__name__ == name:
            return scheduler

    raise ValueError(f'{name} not found')
