from pathlib import Path

from ..configer import ModelConfiger
from .engine import Engine

__all__ = [
    'Engine',
    'lazy_engine',
]


def lazy_engine(configer: ModelConfiger) -> Engine:
    weight_path = Path(configer.weight_path)

    if not weight_path.is_file():
        raise FileNotFoundError(f'Weight file `{weight_path}` not found.')

    suffix = weight_path.suffix

    if suffix == '.onnx':
        from .ort_engine import OrtEngine

        return OrtEngine.load(configer)

    raise NotImplementedError(
        f'Unsupported weight format: `{suffix}`, path: {weight_path}'
    )
