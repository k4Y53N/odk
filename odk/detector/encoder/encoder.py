from abc import ABC, abstractmethod
from typing import Generic, Sequence

from numpy.typing import NDArray

from ..engine import Engine
from ..types import OptionT

__all__ = [
    'Encoder',
]


class Encoder(ABC, Generic[OptionT]):
    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine) -> 'Encoder[OptionT]':
        """Create an Encoder from the given inference engine.

        Args:
            engine (Engine): The inference engine providing input shape and dtype
                information used to configure preprocessing.

        Returns:
            Encoder[OptionT]: An initialized encoder ready to preprocess inputs.
        """

    @abstractmethod
    def encode(
        self,
        origin_input: Sequence[NDArray],
        option: OptionT,
    ) -> Sequence[NDArray]:
        """Preprocess raw inputs into tensors suitable for model inference.

        Args:
            origin_input (Sequence[NDArray]): A sequence of raw input arrays to be
                encoded.
            option (OptionT): Configuration options controlling the encoding behavior

        Returns:
            Sequence[NDArray]: A sequence of preprocessed arrays matching the model's
                expected input shapes and dtypes.
        """
