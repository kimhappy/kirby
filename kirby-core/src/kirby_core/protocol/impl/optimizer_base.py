from abc import ABC, abstractmethod
import torch

from .. _config import _ItemConfig
from ..._util   import _InheritCheck

class OptimizerBase(
    ABC                          ,
    metaclass     = _InheritCheck,
    required_base = torch.optim.Optimizer):
    @abstractmethod
    def __init__(
        self,
        optimizer_config: _ItemConfig,
        params          : torch.Tensor):
        pass
