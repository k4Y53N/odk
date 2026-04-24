from abc import ABC, abstractmethod
from typing import Sequence

from numpy.typing import DTypeLike, NDArray

from ..configer import ModelConfiger

__all__ = [
    'Engine',
]


class Engine(ABC):
    @classmethod
    @abstractmethod
    def from_configer(cls, configer: ModelConfiger) -> 'Engine':
        """Load a model from the given configuration and return an Engine instance.

        Args:
            configer (ModelConfiger): The model configuration specifying the model
                path, parameters, and runtime options.

        Returns:
            Engine: An initialized engine ready for inference.
        """

    @abstractmethod
    def infer(self, input_tensors: Sequence[NDArray]) -> Sequence[NDArray]:
        """Run inference on the loaded model with the given input arrays.

        Args:
            input_tensors (Sequence[NDArray]): A sequence of NumPy arrays, one per
                model input, matching the expected input shapes and dtypes.

        Returns:
            Sequence[NDArray]: A sequence of NumPy arrays containing the model outputs.
        """

    @property
    @abstractmethod
    def input_shapes(self) -> Sequence[Sequence[int]]:
        """The expected shapes of the model inputs.

        Each inner sequence represents the shape of one input tensor, where ``-1``
        indicates a dynamic dimension.

        Returns:
            Sequence[Sequence[int]]: A sequence of shape tuples, one per model input.
        """

    @property
    @abstractmethod
    def input_dtypes(self) -> Sequence[DTypeLike]:
        """The expected data types of the model inputs.

        Each element corresponds to the NumPy dtype required for the respective input
        array (e.g. ``np.float32``).

        Returns:
            Sequence[DTypeLike]: A sequence of dtypes, one per model input.
        """
