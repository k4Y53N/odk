from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'VideoConvert',
    'VideoScale',
    'VideoConvertScale',
]


@dataclass(slots=True)
class VideoConvert(GstElement): ...


@dataclass(slots=True)
class VideoScale(GstElement): ...


@dataclass(slots=True)
class VideoConvertScale(GstElement): ...
