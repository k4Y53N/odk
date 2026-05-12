from typing import TypeVar

from .configer import ModelConfiger

__all__ = [
    'ConfigT',
    'ParamsT',
    'ResultT',
]

ConfigT = TypeVar('ConfigT', bound=ModelConfiger)
ParamsT = TypeVar('ParamsT')
ResultT = TypeVar('ResultT')
