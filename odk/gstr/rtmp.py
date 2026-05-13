from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'RtmpSrc',
    'RtmpSink',
    'Rtmp2Src',
    'Rtmp2Sink',
]


@dataclass
class RtmpSrc(GstElement):
    location: str


@dataclass
class RtmpSink(GstElement):
    location: str


@dataclass
class Rtmp2Src(GstElement):
    location: str


@dataclass
class Rtmp2Sink(GstElement):
    location: str
