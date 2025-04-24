from typing import Type

from .  lstm     import LSTM
from ...protocol import ModelBase

_MODELS = [
    LSTM
]

def register_model(model: Type[ModelBase]) -> None:
    _MODELS.append(model)

def _find_model(name: str) -> Type[ModelBase]:
    for model in _MODELS:
        if model.__name__ == name:
            return model

    raise ValueError(f'{name} not found')
