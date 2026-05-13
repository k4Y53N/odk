from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'QTMux',
    'QTDemux',
    'MP4Mux',
]


@dataclass(slots=True)
class QTMux(GstElement): ...


@dataclass(slots=True)
class QTDemux(GstElement): ...


@dataclass(slots=True)
class MP4Mux(GstElement): ...
