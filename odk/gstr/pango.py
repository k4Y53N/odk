from dataclasses import dataclass
from typing import Literal

from .element import GstElement
from .util import get_numer_denom_str

__all__ = [
    'TextOverlay',
    'ClockOverlay',
    'TimeOverlay',
]


@dataclass
class TextOverlay(GstElement):
    text: str | None = None
    color: int | None = None
    auto_resize: bool | None = None
    scale_mode: Literal['none', 'par', 'display', 'user'] | None = None
    scale_pixel_aspect_ratio: int | float | str | None = None
    halignment: (
        Literal[
            'left',
            'center',
            'right',
            'Absolute position clamped to canvas',
            'Absolute position',
        ]
        | None
    ) = None
    valignment: (
        Literal[
            'baseline',
            'bottom',
            'top',
            'Absolute position clamped to canvas',
            'center',
            'Absolute position',
        ]
        | None
    ) = None
    xpad: int | None = None
    ypad: int | None = None
    xpos: float | None = None
    ypos: float | None = None
    x_absolute: float | None = None
    y_absolute: float | None = None
    shaded_background: bool | None = None
    shading_value: int | None = None

    def __post_init__(self):
        if not isinstance(self.scale_pixel_aspect_ratio, (int, float)):
            return

        self.scale_pixel_aspect_ratio = get_numer_denom_str(
            self.scale_pixel_aspect_ratio
        )


@dataclass
class ClockOverlay(TextOverlay):
    time_format: str | None = None


@dataclass
class TimeOverlay(TextOverlay):
    time_mode: (
        Literal[
            'buffer-time',
            'stream-time',
            'running-time',
            'time-code',
            'elapsed-running-time',
        ]
        | None
    ) = None
