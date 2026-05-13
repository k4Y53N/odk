from dataclasses import dataclass

from .element import GstElement, RawElement
from .util import get_numer_denom_str

__all__ = [
    'RawCaps',
    'VideoCaps',
    'RawVideoCaps',
    'RawVideoBGRCaps',
]


class RawCaps(RawElement):
    def __init__(self, E: str, **kwargs):
        super().__init__(E, **kwargs)

    def separate(self):
        return ','


@dataclass
class VideoCaps(GstElement):
    width: int | None = None
    height: int | None = None
    framerate: int | float | str | None = None
    format: str | None = None

    def __post_init__(self):
        if isinstance(self.framerate, (int, float)):
            self.framerate = get_numer_denom_str(self.framerate)

    def separate(self):
        return ','


@dataclass
class RawVideoCaps(VideoCaps):
    def T(self) -> str:
        return 'video/x-raw'


@dataclass
class RawVideoBGRCaps(RawVideoCaps):
    format: str = 'BGR'
