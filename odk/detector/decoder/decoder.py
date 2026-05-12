from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from numpy.typing import NDArray

from ..engine import Engine
from ..types import ParamsT, ResultT

__all__ = [
    'Decoder',
]


class Decoder(ABC, Generic[ParamsT, ResultT]):
    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine) -> 'Decoder[ParamsT, ResultT]':
        """Load and initialize a decoder from the given engine.

        Args:
            engine (Engine): The inference engine used to retrieve model output
                metadata for configuring the decoder.

        Returns:
            Decoder[ParamsT, ResultT]: A configured decoder instance ready to decode
                model outputs.
        """

    @abstractmethod
    def decode(
        self,
        origin_input: Sequence[NDArray],
        model_output: Sequence[NDArray],
        params: ParamsT,
    ) -> ResultT:
        """Decode raw model outputs into structured results.

        Args:
            origin_input (Sequence[NDArray]): The original inputs before preprocessing.
            model_output (Sequence[NDArray]): The raw output tensors from the inference engine.
            params (ParamsT): Params that control the decoding behavior.

        Returns:
            ResultT: The decoded results.
        """
