from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'FlvMux',
    'FlvDemux',
]


@dataclass
class FlvMux(GstElement): ...


@dataclass
class FlvDemux(GstElement): ...
