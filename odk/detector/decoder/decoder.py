from abc import ABC, abstractmethod
from typing import Generic, Sequence

from numpy.typing import NDArray

from ..engine import Engine
from ..types import OptionT, ResultT

__all__ = [
    'Decoder',
]


class Decoder(ABC, Generic[OptionT, ResultT]):
    @classmethod
    @abstractmethod
    def load(cls, engine: Engine) -> 'Decoder[OptionT, ResultT]':
        """Load and initialize a decoder from the given engine.

        Args:
            engine (Engine): The inference engine used to retrieve model output
                metadata for configuring the decoder.

        Returns:
            Decoder[OptionT, ResultT]: A configured decoder instance ready to decode
                model outputs.
        """

    @abstractmethod
    def decode(
        self,
        origin_input: Sequence[NDArray],
        model_output: Sequence[NDArray],
        option: OptionT,
    ) -> ResultT:
        """Decode raw model outputs into structured results.

        Args:
            origin_input (Sequence[NDArray]): The original inputs before preprocessing.
            model_output (Sequence[NDArray]): The raw output tensors from the inference engine.
            option (OptionT): Options that control the decoding behavior.

        Returns:
            ResultT: The decoded results.
        """
