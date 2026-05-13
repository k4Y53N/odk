from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'QTMux',
    'QTDemux',
    'MP4Mux',
]


@dataclass
class QTMux(GstElement): ...


@dataclass
class QTDemux(GstElement): ...


@dataclass
class MP4Mux(GstElement): ...
