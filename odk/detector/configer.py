import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = [
    'ModelConfiger',
    'Version',
    'ObjectDetectConfiger',
]


@dataclass
class ModelConfiger:
    weight_path: str

    @classmethod
    def from_config_path(cls, path: str):
        with open(path, 'r') as f:
            fields: dict[str, Any] = json.load(f)

        return cls(**fields)


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
