from typing import Literal

import gstr as g


def rtsp_to_app(
    location: str,
    protocols: Literal['tcp', 'udp', 'udp-mcast'] | None = 'tcp',
    fps: int | float | str | None = None,
) -> str:
    pipeline = (
        g.RtspSrc(location=location, protocols=protocols)
        | g.DecodeBin()
        | g.VideoConvert()
    )

    if fps is not None:
        pipeline | g.VideoRate(drop_only=True)

    pipeline | g.RawVideoBGRCaps(framerate=fps) | g.AppSink(drop=True)

    return pipeline.build()


def rtmp_to_app(
    location: str,
    fps: int | float | str | None = None,
) -> str:
    pipeline = g.RtmpSrc(location=location) | g.DecodeBin() | g.VideoConvert()

    if fps is not None:
        pipeline | g.VideoRate(drop_only=True)

    pipeline | g.RawVideoBGRCaps(framerate=fps) | g.AppSink(drop=True)

    return pipeline.build()


def file_to_app(location: str) -> str:
    return (
        g.FileSrc(location=location)
        | g.DecodeBin()
        | g.VideoConvert()
        | g.RawVideoBGRCaps()
        | g.AppSink()
    ).build()
