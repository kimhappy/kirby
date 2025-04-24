from abc import ABC, abstractmethod
import torch

from .. _config import _GeneralConfig, _ItemConfig
from ..._util   import _InheritCheck

class ModelBase(
    ABC                          ,
    metaclass     = _InheritCheck,
    required_base = torch.nn.Module):
    @abstractmethod
    def __init__(
        self                         ,
        general_config: _GeneralConfig,
        model_config  : _ItemConfig):
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor) -> torch.Tensor:
        pass
