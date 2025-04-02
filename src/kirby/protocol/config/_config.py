from typing import Protocol, List

from ._general_config import _GeneralConfig
from ._train_config   import _TrainConfig
from ._item_config    import _ItemConfig
from ._data           import _Data

class _Config(Protocol):
    general   : _GeneralConfig
    train     : _TrainConfig
    model     : _ItemConfig
    trainer   : _ItemConfig
    train_loss: _ItemConfig
    vali_loss : _ItemConfig
    optimizer : _ItemConfig
    scheduler : _ItemConfig
    train_data: List[_Data]
    vali_data : List[_Data]
