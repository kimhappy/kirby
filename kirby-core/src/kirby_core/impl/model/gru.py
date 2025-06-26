import torch

from ...protocol import _GeneralConfig, _ItemConfig, ModelBase

class GRU(
    ModelBase,
    torch.nn.Module):
    def __init__(
        self                          ,
        general_config: _GeneralConfig,
        model_config  : _ItemConfig):
        ModelBase      .__init__(self, general_config, model_config)
        torch.nn.Module.__init__(self)

        self.rec = torch.nn.GRU(
            general_config.num_cond + 1,
            model_config.hidden        ,
            batch_first = True)
        self.lin = torch.nn.Linear(
            model_config.hidden,
            1)
        self.hidden = None

    def forward(self, x):
        x = x.contiguous()
        rec_out, self.hidden = self.rec(x, self.hidden)
        lin_out = self.lin(rec_out)
    
        base = x[..., 0].contiguous()
        delta = lin_out[..., 0].contiguous()
        return base + delta

    def detach(self):
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def reset(self):
        self.hidden = None
