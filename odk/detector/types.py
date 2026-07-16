from typing import TypeVar

from .configer import ModelConfiger

__all__ = [
    'ConfigT',
    'ParamsT',
    'InputT',
    'ResultT',
]

ConfigT = TypeVar('ConfigT', bound=ModelConfiger)
ParamsT = TypeVar('ParamsT')
InputT = TypeVar('InputT')
ResultT = TypeVar('ResultT')
