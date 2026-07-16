from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from numpy.typing import NDArray

from ..engine import Engine
from ..types import InputT, ParamsT, ResultT

__all__ = [
    'Decoder',
]


class Decoder(ABC, Generic[InputT, ResultT, ParamsT]):
    @classmethod
    @abstractmethod
    def from_engine(cls, engine: Engine) -> 'Decoder[InputT, ResultT, ParamsT]':
        """Load and initialize a decoder from the given engine.

        Args:
            engine (Engine): The inference engine used to retrieve model output
                metadata for configuring the decoder.

        Returns:
            Decoder[InputT, ResultT, ParamsT]: A configured decoder instance ready to
                decode model outputs.
        """

    @abstractmethod
    def decode(
        self,
        input: InputT,
        output: Sequence[NDArray],
        params: ParamsT,
    ) -> ResultT:
        """Decode raw model outputs into structured results.

        Args:
            input (InputT): The original inputs before preprocessing.
            output (Sequence[NDArray]): The raw output tensors from the inference engine.
            params (ParamsT): Params that control the decoding behavior.

        Returns:
            ResultT: The decoded results.
        """
