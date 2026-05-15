import cv2

from .image import Image

__all__ = [
    'Video',
    'VideoWriter',
]


class Video:
    """A wrapper around OpenCV VideoCapture for reading video frames."""

    def __init__(self, source: str, api: int):
        """Initialize a Video instance.

        Args:
            source (str): The video source.
            api (int): The OpenCV backend API identifier.
        """
        self.__capture = cv2.VideoCapture(source, api)
        self.__source: str = source
        self.__api: int = api

    def __len__(self):
        """Return the number of frames in the video."""
        return max(self.frame_count, 0)

    def __iter__(self):
        """Return the iterator object (self)."""
        return self

    def __next__(self) -> Image:
        """Return the next frame, or raise StopIteration when exhausted."""
        if image := self.read():
            return image

        raise StopIteration()

    @classmethod
    def from_uri(cls, source: str) -> 'Video':
        """Create a Video instance from a URI or file path.

        Args:
            source (str): The URI or file path of the video source.

        Returns:
            Video: A new Video instance using the default backend.
        """
        return cls(source, cv2.CAP_ANY)

    @classmethod
    def from_gst(cls, pipeline: str) -> 'Video':
        """Create a Video instance from a GStreamer pipeline.

        Args:
            pipeline (str): The GStreamer pipeline description string.

        Returns:
            Video: A new Video instance using the GStreamer backend.
        """
        return cls(pipeline, cv2.CAP_GSTREAMER)

    @property
    def size(self) -> tuple[int, int]:
        """Video size ``(height, width)``

        Returns:
            tuple[int, int]: ``(height, width)``
        """
        return self.height, self.width

    @property
    def width(self) -> int:
        """Video width.

        Returns:
            int: Width.
        """
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Video height.

        Returns:
            int: Height.
        """
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """Video FPS.

        Returns:
            float: FPS.
        """
        return self.__capture.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        """Total number of frames in the video.

        Returns:
            int: The total frame count.
        """
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Image | None:
        """Read the next frame from the video.

        Returns:
            Image | None: The next frame as an Image, or None if no frame is available.
        """
        ret, frame = self.__capture.read()

        if not ret:
            return None

        return Image(data=frame)

    def open(self) -> bool:
        """Open or reopen the video capture.

        Returns:
            bool: True if the video capture was successfully opened.
        """
        self.release()
        return self.__capture.open(self.__source, self.__api)

    def is_opened(self) -> bool:
        """Check whether the video capture is currently open.

        Returns:
            bool: True if the video capture is open.
        """
        return self.__capture.isOpened()

    def release(self):
        """Release the video capture and free associated resources."""
        self.__capture.release()

    def seek(self, offset: int):
        """Seek to a specific frame position in the video.

        Args:
            offset (int): The target frame index to seek to.
        """
        self.__capture.set(cv2.CAP_PROP_POS_FRAMES, offset)

    def tell(self) -> int:
        """Get the current frame position in the video.

        Returns:
            int: The current frame index.
        """
        return int(self.__capture.get(cv2.CAP_PROP_POS_FRAMES))


class VideoWriter:
    def __init__(self, writer: cv2.VideoWriter, width: int, height: int):
        self.__writer: cv2.VideoWriter = writer
        self.__width: int = width
        self.__height: int = height

    @classmethod
    def to_file(
        cls,
        filename: str,
        fourcc: str,
        fps: float,
        width: int,
        height: int,
    ) -> 'VideoWriter':
        """Create a VideoWriter that writes to a file.

        Args:
            filename (str): The output file path.
            fourcc (str): A 4-character codec code (e.g. ``'mp4v'``, ``'XVID'``).
            fps (float): The frame rate of the output video.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        fourcc_code = cv2.VideoWriter.fourcc(*fourcc)
        writer = cv2.VideoWriter(
            filename=filename,
            apiPreference=cv2.CAP_ANY,
            fourcc=fourcc_code,
            fps=fps,
            frameSize=(width, height),
        )

        return cls(writer, width=width, height=height)

    @classmethod
    def to_file_like(cls, filename: str, fourcc: str, video: Video) -> 'VideoWriter':
        """Create a VideoWriter that writes to a file, using a Video's properties.

        Args:
            filename (str): The output file path.
            fourcc (str): A 4-character codec code (e.g. ``'mp4v'``, ``'XVID'``).
            video (Video): The source video to copy fps, width, and height from.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        return cls.to_file(
            filename=filename,
            fourcc=fourcc,
            fps=video.fps,
            width=video.width,
            height=video.height,
        )

    @classmethod
    def to_mp4_file(
        cls,
        filename: str,
        fps: float,
        width: int,
        height: int,
    ) -> 'VideoWriter':
        """Create a VideoWriter that writes to an MP4 file.

        Args:
            filename (str): The output file path.
            fps (float): The frame rate of the output video.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        return cls.to_file(
            filename=filename,
            fourcc='mp4v',
            fps=fps,
            width=width,
            height=height,
        )

    @classmethod
    def to_mp4_file_like(cls, filename: str, video: Video) -> 'VideoWriter':
        """Create a VideoWriter that writes to an MP4 file, using a Video's properties.

        Args:
            filename (str): The output file path.
            video (Video): The source video to copy fps, width, and height from.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        return cls.to_file_like(filename=filename, fourcc='mp4v', video=video)

    @classmethod
    def to_gst(
        cls,
        pipeline: str,
        fps: float,
        width: int,
        height: int,
    ) -> 'VideoWriter':
        """Create a VideoWriter that writes to a GStreamer pipeline.

        Args:
            pipeline (str): The GStreamer pipeline description string.
            fps (float): The frame rate of the output video.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        writer = cv2.VideoWriter(
            filename=pipeline,
            apiPreference=cv2.CAP_GSTREAMER,
            fourcc=0,
            fps=fps,
            frameSize=(width, height),
        )
        return cls(writer, width, height)

    @classmethod
    def to_gst_like(cls, pipeline: str, video: Video) -> 'VideoWriter':
        """Create a VideoWriter that writes to a GStreamer pipeline, using a Video's
        properties.

        Args:
            pipeline (str): The GStreamer pipeline description string.
            video (Video): The source video to copy fps, width, and height from.

        Returns:
            VideoWriter: A new VideoWriter instance.
        """
        return cls.to_gst(
            pipeline=pipeline,
            fps=video.fps,
            width=video.width,
            height=video.height,
        )

    def write(self, image: Image):
        """Write a frame to the video output, resizing if necessary.

        Args:
            image (Image): The frame to write.
        """
        data = image.data

        if data.shape[0] != self.__height or data.shape[1] != self.__width:
            data = cv2.resize(data, (self.__width, self.__height))

        self.__writer.write(data)

    def is_opened(self) -> bool:
        """Check whether the video writer is currently open.

        Returns:
            bool: True if the video writer is open.
        """
        return self.__writer.isOpened()

    def release(self):
        """Release the video writer and free associated resources."""
        self.__writer.release()
