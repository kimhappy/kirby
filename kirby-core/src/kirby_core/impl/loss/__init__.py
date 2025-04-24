from typing import Type
import torch

from .  dc       import DC
from .  esr      import ESR
from ...protocol import _ItemConfig, LossBase

class Mix(
    LossBase,
    torch.nn.Module):
    def __init__(
        self,
        loss_config: _ItemConfig):
        LossBase       .__init__(self, loss_config)
        torch.nn.Module.__init__(self)

        self.losses = []

        for elem_config in loss_config.elems:
            elem = _find_loss(elem_config.name)
            elem = elem(elem_config)
            self.losses.append([elem, elem_config.ratio])

    def forward(
        self                ,
        output: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
        ret = 0.0

        for loss, weight in self.losses:
            ret += weight * loss(output, target)

        return ret

_LOSSES = [
    Mix,
    DC ,
    ESR
]

def register_loss(loss: Type[LossBase]) -> None:
    _LOSSES.append(loss)

def _find_loss(name: str) -> Type[LossBase]:
    for loss in _LOSSES:
        if loss.__name__ == name:
            return loss

    raise ValueError(f'{name} not found')
