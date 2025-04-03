import torch

from ...protocol._config._item_config import _ItemConfig
from ...protocol.impl   .loss_base    import LossBase

class DC(
    LossBase,
    torch.nn.Module):
    _EPSILON = 0.00001

    def __init__(
        self,
        loss_config: _ItemConfig):
        LossBase       .__init__(self, loss_config)
        torch.nn.Module.__init__(self)

    def forward(
        self                ,
        output: torch.Tensor,
        target: torch.Tensor) -> torch.Tensor:
        loss   = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(output, 0)), 2)
        loss   = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2))
        loss   = torch.div(loss, energy + self._EPSILON)
        return loss
