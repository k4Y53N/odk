from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from numpy.typing import NDArray

from ..engine import Engine
from ..types import InputT, ParamsT

__all__ = [
    'Encoder',
]


class Encoder(ABC, Generic[InputT, ParamsT]):
    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine) -> 'Encoder[InputT, ParamsT]':
        """Create an Encoder from the given inference engine.

        Args:
            engine (Engine): The inference engine providing input shape and dtype
                information used to configure preprocessing.

        Returns:
            Encoder[InputT, ParamsT]: An initialized encoder ready to preprocess
                inputs.
        """

    @abstractmethod
    def encode(self, input: InputT, params: ParamsT) -> Sequence[NDArray]:
        """Preprocess raw inputs into tensors suitable for model inference.

        Args:
            input (InputT): Raw input to be encoded.
            params (ParamsT): Params controlling the encoding behavior.

        Returns:
            Sequence[NDArray]: A sequence of preprocessed arrays matching the model's
                expected input shapes and dtypes.
        """
