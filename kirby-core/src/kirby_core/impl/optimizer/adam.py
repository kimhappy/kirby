import torch

from ...protocol._config._item_config   import _ItemConfig
from ...protocol.impl   .optimizer_base import OptimizerBase

class Adam(
    OptimizerBase,
    torch.optim.Adam):
    def __init__(
        self,
        optimizer_config: _ItemConfig,
        params          : torch.Tensor):
        OptimizerBase   .__init__(
            self            ,
            optimizer_config,
            params)
        torch.optim.Adam.__init__(
            self,
            params       = params                     ,
            lr           = optimizer_config.initial_lr,
            weight_decay = optimizer_config.lr_decay)
