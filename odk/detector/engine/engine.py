from abc import ABC, abstractmethod
from numpy.typing import NDArray, DTypeLike
from ..configer import ModelConfiger


class Engine(ABC):
    @classmethod
    @abstractmethod
    def load(cls, configer: ModelConfiger) -> 'Engine':
        """Load a model from the given configuration and return an Engine instance.

        Args:
            configer (ModelConfiger): The model configuration specifying the model
                path, parameters, and runtime options.

        Returns:
            Engine: An initialized engine ready for inference.
        """

    @abstractmethod
    def infer(self, input_tensors: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
        """Run inference on the loaded model with the given input arrays.

        Args:
            input_tensors (tuple[NDArray, ...]): A tuple of NumPy arrays, one per model
                input, matching the expected input shapes and dtypes.

        Returns:
            tuple[NDArray, ...]: A tuple of NumPy arrays containing the model outputs.
        """

    @property
    @abstractmethod
    def input_shapes(self) -> tuple[tuple[int | None, ...], ...]:
        """The expected shapes of the model inputs.

        Each inner tuple represents the shape of one input, where ``None``
        indicates a dynamic dimension (e.g. batch size).

        Returns:
            tuple[tuple[int | None, ...], ...]: A tuple of shape tuples, one per model input.
        """

    @property
    @abstractmethod
    def input_dtypes(self) -> tuple[DTypeLike, ...]:
        """The expected data types of the model inputs.

        Each element corresponds to the NumPy dtype required for the respective input
        array (e.g. ``np.float32``).

        Returns:
            tuple[DTypeLike, ...]: A tuple of dtypes, one per model input.
        """
