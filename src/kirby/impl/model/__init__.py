from typing import Type

from ...protocol.impl.model_base import ModelBase
from .  lstm                     import LSTM

_MODELS = [
    LSTM
]

def register_model(model: Type[ModelBase]) -> None:
    _MODELS.append(model)

def _find_model(name: str) -> Type[ModelBase]:
    for model in _MODELS:
        if model.__name__ == name:
            return model

    raise ValueError(f'{ name } not found')

__all__ = [
    'register_model',
    '_find_model'
]
