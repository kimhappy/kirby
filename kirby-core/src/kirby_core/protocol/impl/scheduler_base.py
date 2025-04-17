from abc import ABC, abstractmethod
import torch

from .  optimizer_base       import OptimizerBase
from .. _config._item_config import _ItemConfig
from ..._util  ._meta        import _InheritCheck

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
