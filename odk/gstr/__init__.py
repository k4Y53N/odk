from dataclasses import dataclass

from .app import *  # noqa
from .caps import *  # noqa
from .core import *  # noqa
from .element import *  # noqa
from .flv import *  # noqa
from .h264 import *  # noqa
from .isomp4 import *  # noqa
from .pango import *  # noqa
from .playback import *  # noqa
from .rtmp import *  # noqa
from .rtp import *  # noqa
from .rtsp import *  # noqa
from .videoconvertscale import *  # noqa


@dataclass(slots=True)
class VideoRate(GstElement):  # noqa
    drop_only: bool | None = None


@dataclass(slots=True)
class VideoTestSrc(GstElement): ...  # noqa


@dataclass(slots=True)
class AutoVideoSink(GstElement): ...  # noqa
