import torch

from ...protocol._config._general_config import _GeneralConfig
from ...protocol._config._item_config    import _ItemConfig
from ...protocol.impl   .model_base      import ModelBase

class LSTM(
    ModelBase,
    torch.nn.Module):
    def __init__(
        self                          ,
        general_config: _GeneralConfig,
        model_config  : _ItemConfig):
        ModelBase      .__init__(self, general_config, model_config)
        torch.nn.Module.__init__(self)

        self.rec = torch.nn.LSTM(
            general_config.num_cond + 1,
            model_config.hidden        ,
            batch_first = True)
        self.lin = torch.nn.Linear(
            model_config.hidden,
            1)
        self.hidden = None

    def forward(self, x):
        rec_out, self.hidden = self.rec(x, self.hidden)
        lin_out              = self.lin(rec_out)
        return x[ ..., 0 ] + lin_out[ ..., 0 ]

    def detach(self):
        self.hidden = tuple([h.clone().detach() for h in self.hidden])

    def reset(self):
        self.hidden = None
