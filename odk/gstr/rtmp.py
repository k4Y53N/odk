from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'RtmpSrc',
    'RtmpSink',
    'Rtmp2Src',
    'Rtmp2Sink',
]


@dataclass(slots=True)
class RtmpSrc(GstElement):
    location: str


@dataclass(slots=True)
class RtmpSink(GstElement):
    location: str


@dataclass(slots=True)
class Rtmp2Src(GstElement):
    location: str


@dataclass(slots=True)
class Rtmp2Sink(GstElement):
    location: str
