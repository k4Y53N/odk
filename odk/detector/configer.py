from dataclasses import dataclass
from enum import Enum

__all__ = [
    'ModelConfiger',
    'Version',
    'ObjectDetectConfiger',
]


@dataclass
class ModelConfiger:
    weight_path: str


class Version(str, Enum):
    V4 = 'v4'
    V7 = 'v7'
    V8 = 'v8'
    V9 = 'v9'
    V11 = 'v11'

    def __str__(self) -> str:
        return self.value


@dataclass
class ObjectDetectConfiger(ModelConfiger):
    version: Version
    class_label: list[str]
