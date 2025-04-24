import torch

from ...protocol import _ItemConfig, OptimizerBase

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
