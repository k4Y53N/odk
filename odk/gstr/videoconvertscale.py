from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'VideoConvert',
    'VideoScale',
    'VideoConvertScale',
]


@dataclass
class VideoConvert(GstElement): ...


@dataclass
class VideoScale(GstElement): ...


@dataclass
class VideoConvertScale(GstElement): ...
