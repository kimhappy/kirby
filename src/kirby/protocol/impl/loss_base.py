from abc import ABC, abstractmethod
import torch

from .. config._item_config import _ItemConfig
from ...util  ._meta        import _InheritCheck

class LossBase(
    ABC                          ,
    metaclass     = _InheritCheck,
    required_base = torch.nn.Module):
    @abstractmethod
    def __init__(
        self,
        loss_config: _ItemConfig):
        pass

    @abstractmethod
    def forward(
        self                ,
        output: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
        pass
