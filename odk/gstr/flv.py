from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'FlvMux',
    'FlvDemux',
]


@dataclass(slots=True)
class FlvMux(GstElement): ...


@dataclass(slots=True)
class FlvDemux(GstElement): ...
