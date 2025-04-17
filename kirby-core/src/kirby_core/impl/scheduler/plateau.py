import torch

from ...protocol._config._item_config   import _ItemConfig
from ...protocol.impl   .optimizer_base import OptimizerBase
from ...protocol.impl   .scheduler_base import SchedulerBase

class Plateau(
    SchedulerBase,
    torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(
        self                         ,
        scheduler_config: _ItemConfig,
        optimizer       : OptimizerBase):
        SchedulerBase                             .__init__(
            self            ,
            scheduler_config,
            optimizer)
        torch.optim.lr_scheduler.ReduceLROnPlateau.__init__(
            self                                 ,
            optimizer                            ,
            mode     = 'min'                     ,
            factor   = scheduler_config.lr_factor,
            patience = scheduler_config.lr_patience)
