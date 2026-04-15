from typing import Sequence

import numpy as np
import onnxruntime as ort
from numpy.typing import DTypeLike, NDArray

from ..configer import ModelConfiger
from .engine import Engine

__all__ = [
    'OrtEngine',
]

DTYPE_MAP: dict[str, DTypeLike] = {
    'tensor(float)': np.float32,
    'tensor(float16)': np.float16,
    'tensor(uint8)': np.uint8,
}


class OrtEngine(Engine):
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_names = tuple[str](
            model_input.name for model_input in self.session.get_inputs()
        )

    @classmethod
    def load(cls, configer: ModelConfiger) -> 'OrtEngine':
        return cls(
            ort.InferenceSession(
                path_or_bytes=configer.weight_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            )
        )

    def infer(self, input_tensors: Sequence[NDArray]) -> Sequence[NDArray]:
        input_feed = {
            input_name: input_tensor
            for input_name, input_tensor in zip(self.input_names, input_tensors)
        }

        return tuple(self.session.run(None, input_feed))

    @property
    def input_shapes(self) -> Sequence[Sequence[int]]:
        return tuple(
            tuple(
                shape if isinstance(shape, int) else -1 for shape in model_input.shape
            )
            for model_input in self.session.get_inputs()
        )

    @property
    def input_dtypes(self) -> Sequence[DTypeLike]:
        return tuple(
            DTYPE_MAP[model_input.type] for model_input in self.session.get_inputs()
        )
