from typing import Sequence

import cv2
import numpy as np
from numpy.typing import DTypeLike, NDArray

from ..engine import Engine
from ..option import ObjectDetectOption
from .encoder import Encoder

__all__ = [
    'ImageEncoder',
]


class ImageEncoder(Encoder[ObjectDetectOption]):
    def __init__(self, height: int, width: int, dtype: DTypeLike):
        self.height: int = height
        self.width: int = width
        self.dtype: DTypeLike = dtype

    @classmethod
    def load(cls, engine: Engine) -> 'ImageEncoder':
        input_shape = engine.input_shapes[0]  # [batch, channel, height, width]
        input_dtype = engine.input_dtypes[0]
        height, width = input_shape[2], input_shape[3]

        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f'[height, width] is not int: [{height}, {width}]')

        return cls(height, width, input_dtype)

    def encode(
        self,
        origin_input: Sequence[NDArray[np.uint8]],
        option: ObjectDetectOption,
    ) -> Sequence[NDArray]:
        tensor = [
            cv2.resize(image, (self.width, self.height)) for image in origin_input
        ]
        tensor = np.stack(tensor)
        tensor = tensor[..., ::-1]  # BGR -> RGB
        tensor = np.transpose(tensor, (0, 3, 1, 2))  # [batch, channel, height, width]

        if self.dtype is np.uint8:
            return [tensor]

        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        tensor /= 255.0

        if tensor.dtype is not self.dtype:
            tensor = np.ascontiguousarray(tensor, self.dtype)

        return [tensor]
