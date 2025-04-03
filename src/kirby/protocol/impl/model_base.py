from abc import ABC, abstractmethod
import torch

from .. _config._general_config import _GeneralConfig
from .. _config._item_config    import _ItemConfig
from ..._util  ._meta           import _InheritCheck

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
