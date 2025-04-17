from typing import List
from abc    import ABC, abstractmethod
import torch

from . model_base              import ModelBase
from . loss_base               import LossBase
from . optimizer_base          import OptimizerBase
from . scheduler_base          import SchedulerBase
from . result                  import Result
from .._config._general_config import _GeneralConfig
from .._config._item_config    import _ItemConfig
from .._config._data           import _Data

class TrainerBase(ABC):
    @abstractmethod
    def __init__(
        self                          ,
        general_config: _GeneralConfig,
        trainer_config: _ItemConfig   ,
        model         : ModelBase     ,
        train_loss    : LossBase      ,
        vali_loss     : LossBase      ,
        optimizer     : OptimizerBase ,
        scheduler     : SchedulerBase ,
        train_data    : List[_Data]   ,
        vali_data     : List[_Data]   ,
        device        : torch.device):
        self.num_cond    = general_config.num_cond
        self.sample_rate = general_config.sample_rate
        self.model       = model
        self.train_loss  = train_loss
        self.vali_loss   = vali_loss
        self.optimizer   = optimizer
        self.scheduler   = scheduler
        self.device      = device

    @abstractmethod
    def train(self) -> Result:
        pass

    @abstractmethod
    def validate(self) -> Result:
        pass
