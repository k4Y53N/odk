from dataclasses import dataclass
from typing import Literal

from .element import GstElement

__all__ = [
    'RtspSrc',
    'RtspClientSink',
]


@dataclass
class RtspSrc(GstElement):
    location: str
    protocols: Literal['tcp', 'udp', 'udp-mcast' 'tcp+udp-mcast+udp'] | None = None


@dataclass
class RtspClientSink(GstElement):
    location: str
    protocols: Literal['tcp', 'udp', 'udp-mcast' 'tcp+udp-mcast+udp'] | None = None
