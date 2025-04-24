from abc import ABC, abstractmethod
import torch

from .  optimizer_base import OptimizerBase
from .. _config        import _ItemConfig
from ..._util          import _InheritCheck

class SchedulerBase(
    ABC                          ,
    metaclass     = _InheritCheck,
    required_base = torch.optim.lr_scheduler.LRScheduler):
    @abstractmethod
    def __init__(
        self                         ,
        scheduler_config: _ItemConfig,
        optimizer       : OptimizerBase):
        pass
