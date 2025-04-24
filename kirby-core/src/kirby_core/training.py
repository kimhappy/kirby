from __future__ import annotations

from typing import Callable, Tuple
from copy   import deepcopy
import box
import torch

from .protocol import _Config, Result
from .impl     import _find_model, _find_loss, _find_optimizer, _find_scheduler, _find_trainer
from ._util    import _set_seed, _set_deterministic

def training(
    config  : _Config | dict,
    callback: Callable[[int, Result], None]) -> Tuple[dict, Result]:
    # Initialize
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
    max_epochs = config.train.max_epochs
    early_stop = config.train.early_stop
    vali_cycle = config.train.vali_cycle

    train_loss = _find_loss(config.train_loss.name)
    train_loss = train_loss(config.train_loss)

    vali_loss = _find_loss(config.vali_loss.name)
    vali_loss = vali_loss(config.vali_loss)

    optimizer = _find_optimizer(config.optimizer.name)
    optimizer = optimizer(config.optimizer, model.parameters())

    scheduler = _find_scheduler(config.scheduler.name)
    scheduler = scheduler(config.scheduler, optimizer)

    # Make trainer
    trainer = _find_trainer(config.trainer.name)
    trainer = trainer(
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

    # Run
    best_vali_loss   = 1e12
    best_dict        = None
    patience_counter = 0

    for epoch in range(1, max_epochs + 1):
        tr = trainer.train()
        lr = trainer.optimizer.param_groups[ 0 ][ 'lr' ]

        callback(epoch, tr)
        callback(epoch, Result('learning rate', lr))

        if epoch % vali_cycle != 0:
            continue

        vr = trainer.validate()
        callback(epoch, vr)

        if vr.value < best_vali_loss:
            patience_counter = 0
            best_dict        = deepcopy(trainer.model.state_dict())
            best_vali_loss   = vr.value
            continue

        patience_counter += 1

        if patience_counter > early_stop:
            break

    return best_dict, Result('best validation loss', best_vali_loss)
