from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from numpy.typing import NDArray

from ..engine import Engine
from ..types import ParamsT

__all__ = [
    'Encoder',
]


class Encoder(ABC, Generic[ParamsT]):
    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine) -> 'Encoder[ParamsT]':
        """Create an Encoder from the given inference engine.

        Args:
            engine (Engine): The inference engine providing input shape and dtype
                information used to configure preprocessing.

        Returns:
            Encoder[paramsT]: An initialized encoder ready to preprocess inputs.
        """

    @abstractmethod
    def encode(
        self,
        origin_input: Sequence[NDArray],
        params: ParamsT,
    ) -> Sequence[NDArray]:
        """Preprocess raw inputs into tensors suitable for model inference.

        Args:
            origin_input (Sequence[NDArray]): A sequence of raw input arrays to be
                encoded.
            params (ParamsT): Params controlling the encoding behavior

        Returns:
            Sequence[NDArray]: A sequence of preprocessed arrays matching the model's
                expected input shapes and dtypes.
        """
