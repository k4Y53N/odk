from dataclasses import dataclass
from typing import Literal

from .element import GstElement

__all__ = [
    'RtspSrc',
    'RtspClientSink',
]


@dataclass(slots=True)
class RtspSrc(GstElement):
    location: str
    protocols: Literal['tcp', 'udp', 'udp-mcast', 'tcp+udp-mcast+udp'] | None = None


@dataclass(slots=True)
class RtspClientSink(GstElement):
    location: str
    protocols: Literal['tcp', 'udp', 'udp-mcast' 'tcp+udp-mcast+udp'] | None = None
