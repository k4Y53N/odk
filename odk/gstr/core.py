from dataclasses import dataclass
from typing import Literal

from .element import GstElement

__all__ = [
    'Queue',
    'Tee',
    'FileSrc',
    'FileSink',
    'FDSrc',
    'FDSink',
    'FakeSrc',
    'FakeSink',
]


@dataclass(slots=True)
class Queue(GstElement):
    leaky: Literal['no', 'upstream', 'downstream'] | None = None


@dataclass(slots=True)
class Tee(GstElement): ...


@dataclass(slots=True)
class FileSrc(GstElement):
    location: str


@dataclass(slots=True)
class FileSink(GstElement):
    location: str


@dataclass(slots=True)
class FDSrc(GstElement): ...


@dataclass(slots=True)
class FDSink(GstElement): ...


@dataclass(slots=True)
class FakeSrc(GstElement): ...


@dataclass(slots=True)
class FakeSink(GstElement): ...
