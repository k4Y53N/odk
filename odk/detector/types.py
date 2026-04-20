from typing import TypeVar

from .configer import ModelConfiger

__all__ = [
    'ConfigT',
    'OptionT',
    'ResultT',
]

ConfigT = TypeVar('ConfigT', bound=ModelConfiger)
OptionT = TypeVar('OptionT')
ResultT = TypeVar('ResultT')
