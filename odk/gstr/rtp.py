from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'RtpH264Pay',
    'RtpH264Depay',
]


@dataclass
class RtpH264Pay(GstElement): ...


@dataclass
class RtpH264Depay(GstElement): ...
