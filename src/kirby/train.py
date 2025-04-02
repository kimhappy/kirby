from __future__ import annotations

from typing import Callable, Tuple
from copy   import deepcopy
import box
import time
import torch

from .protocol.config._config import _Config
from .impl                    import _find_model, _find_loss, _find_optimizer, _find_scheduler, _find_trainer
from .util                    import _set_seed, _set_deterministic, Result

class Train:
    def __init__(
        self,
        config: _Config | dict):
        if isinstance(config, dict):
            config = box.Box(config)

        _set_seed(config.general.seed)

        if config.general.deterministic:
            _set_deterministic(config.general.device)

        device = torch.device(config.general.device)

        # Model related
        model = _find_model(config.model.name)
        model = model(config.general, config.model)
        model = model.to(device)

        if config.general.compile:
            model = torch.compile(model)

        # Training related
        self.max_epochs = config.train.max_epochs
        self.early_stop = config.train.early_stop
        self.vali_cycle = config.train.vali_cycle

        train_loss = _find_loss(config.train_loss.name)
        train_loss = train_loss(config.train_loss)

        vali_loss = _find_loss(config.vali_loss.name)
        vali_loss = vali_loss(config.vali_loss)

        optimizer = _find_optimizer(config.optimizer.name)
        optimizer = optimizer(config.optimizer, model.parameters())

        scheduler = _find_scheduler(config.scheduler.name)
        scheduler = scheduler(config.scheduler, optimizer)

        # Make trainer
        trainer      = _find_trainer(config.trainer.name)
        self.trainer = trainer(
            config.general   ,
            config.trainer   ,
            model            ,
            train_loss       ,
            vali_loss        ,
            optimizer        ,
            scheduler        ,
            config.train_data,
            config.vali_data ,
            device)

    def run(self, callback: Callable[[int, Result], None]) -> Tuple[dict, Result]:
        best_vali_loss   = 1e12
        best_dict        = None
        patience_counter = 0
        begin            = time.time()

        for epoch in range(1, self.max_epochs + 1):
            tr = self.trainer.train()
            callback(epoch, tr)

            if epoch % self.vali_cycle != 0:
                continue

            vr = self.trainer.validate()
            callback(epoch, vr)

            if vr.loss < best_vali_loss:
                patience_counter = 0
                best_dict        = deepcopy(self.trainer.model.state_dict())
                best_vali_loss   = vr.loss
                continue

            patience_counter += 1

            if patience_counter > self.early_stop:
                break

        end = time.time()
        return best_dict, Result('final', best_vali_loss, end - begin)
