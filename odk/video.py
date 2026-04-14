import cv2

from .image import Image

__all__ = [
    'Video',
]


class Video:
    """A wrapper around OpenCV VideoCapture for reading video frames."""

    def __init__(self, sorurce: str, api: int):
        """Initialize a Video instance.

        Args:
            sorurce (str): The video source.
            api (int): The OpenCV backend API identifier.
        """
        self.__capture = cv2.VideoCapture(sorurce, api)
        self.source: str = sorurce
        self.api: int = api

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

        self.release()
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
        """Video size [height, width].

        Returns:
            tuple[int, int]: [height, width]
        """
        return self.height, self.width

    @property
    def width(self) -> int:
        """Video width.

        Returns:
            int: width
        """
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Video height.

        Returns:
            int: height
        """
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> int:
        """Video FPS.

        Returns:
            int: FPS
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
        return self.__capture.open(self.source, self.api)

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
