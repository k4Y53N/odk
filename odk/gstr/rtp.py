from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'RtpH264Pay',
    'RtpH264Depay',
]


@dataclass(slots=True)
class RtpH264Pay(GstElement): ...


@dataclass(slots=True)
class RtpH264Depay(GstElement): ...
