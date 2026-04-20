from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar

from numpy.typing import NDArray

from .decoder import Decoder
from .encoder import Encoder
from .engine import lazy_engine
from .types import ConfigT, OptionT, ResultT

__all__ = [
    'Detector',
]

Self = TypeVar('Self', bound='Detector[ConfigT, OptionT, ResultT]')


class Detector(ABC, Generic[ConfigT, OptionT, ResultT]):
    def __init__(self, configer: ConfigT):
        self._engine = lazy_engine(configer)
        self._encoder = self.get_encoder_class(configer).from_engine(self._engine)
        self._decoder = self.get_decoder_class(configer).from_engine(self._engine)
        self._configer = configer

    @classmethod
    @abstractmethod
    def get_configer_class(cls) -> type[ConfigT]:
        """Return the configer class used to load model configuration.

        Returns:
            type[ConfigT]: The configer class capable of parsing a config file into the
                configuration dataclass expected by this detector.
        """

    @classmethod
    @abstractmethod
    def get_encoder_class(cls, configer: ConfigT) -> type[Encoder[OptionT]]:
        """Return the encoder class used for input preprocessing.

        Args:
            configer (ConfigT): The model configuration, which may influence which
                encoder variant is selected.

        Returns:
            type[Encoder[OptionT]]: The encoder class responsible for transforming raw
                inputs into model-ready tensors.
        """

    @classmethod
    @abstractmethod
    def get_decoder_class(cls, configer: ConfigT) -> type[Decoder[OptionT, ResultT]]:
        """Return the decoder class used for output postprocessing.

        Args:
            configer (ConfigT): The model configuration, which may influence which
                decoder variant is selected.

        Returns:
            type[Decoder[OptionT, ResultT]]: The decoder class responsible for
                converting raw model outputs into structured results.
        """

    @classmethod
    def from_config_path(cls: type[Self], path: str) -> Self:
        """Create a detector instance from a configuration file path.

        Loads the model configuration using the configer class returned by
        ``get_configer_class``, then constructs the detector with the resulting
        configuration.

        Args:
            path (str): Path to the model configuration file.

        Returns:
            Self: Detector instance.
        """
        configer = cls.get_configer_class().from_config_path(path)
        return cls(configer)

    def infer(self, origin: Sequence[NDArray], option: OptionT) -> ResultT:
        """Run the full detection pipeline: encode, infer, and decode.

        Preprocesses the raw inputs through the encoder, runs model inference via the
        engine, and postprocesses the outputs through the decoder.

        Args:
            origin (Sequence[NDArray]): Raw input arrays before any preprocessing.
            option (OptionT): Options controlling encoding and decoding behaviour for
                this inference call.

        Returns:
            ResultT: Structured results produced by the decoder.
        """
        input_tensors = self._encoder.encode(origin_input=origin, option=option)
        model_output = self._engine.infer(input_tensors=input_tensors)
        result = self._decoder.decode(
            origin_input=origin,
            model_output=model_output,
            option=option,
        )

        return result
