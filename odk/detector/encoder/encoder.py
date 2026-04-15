from abc import ABC, abstractmethod
from typing import Sequence

from numpy.typing import NDArray

from ..engine import Engine

__all__ = [
    'Encoder',
]


class Encoder(ABC):
    @classmethod
    @abstractmethod
    def load(cls, engine: Engine) -> 'Encoder':
        """Create an Encoder from the given inference engine.

        Args:
            engine (Engine): The inference engine providing input shape and dtype
                information used to configure preprocessing.

        Returns:
            Encoder: An initialized encoder ready to preprocess inputs.
        """

    @abstractmethod
    def encode(self, original_input: Sequence[NDArray]) -> Sequence[NDArray]:
        """Preprocess raw inputs into tensors suitable for model inference.

        Args:
            original_input (Sequence[NDArray]): A sequence of raw input arrays to be
                encoded.

        Returns:
            Sequence[NDArray]: A sequence of preprocessed arrays matching the model's
                expected input shapes and dtypes.
        """
