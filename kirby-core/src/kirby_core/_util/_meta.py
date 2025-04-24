from typing import Any
from abc    import ABCMeta

class _InheritCheck(ABCMeta):
    def __new__(
        mcls                      ,
        name         : str        ,
        bases        : tuple      ,
        namespace    : dict       , *,
        required_base: type = None,
        **kwargs     : Any) -> Any:
        cls = super().__new__(mcls, name, bases, namespace)

        if required_base is not None:
            setattr(cls, '_required_base', required_base)

        return cls

    def __init__(
        cls                       ,
        name         : str        ,
        bases        : tuple      ,
        namespace    : dict       , *,
        required_base: type = None,
        **kwargs     : Any) -> None:
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        required_base = getattr(cls, '_required_base', None)

        if required_base is not None and not any(issubclass(base, required_base) for base in cls.__bases__):
            raise TypeError(f'Classes inheriting from {cls.__name__} must also inherit from {required_base.__name__}')

        return super().__call__(*args, **kwargs)
