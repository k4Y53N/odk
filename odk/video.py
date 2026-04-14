import cv2

from .image import Image

__all__ = [
    'Video',
]


class Video:
    def __init__(self, sorurce: str, api: int):
        self.__capture = cv2.VideoCapture(sorurce, api)
        self.source: str = sorurce
        self.api: int = api

    def __len__(self):
        return max(self.frame_count, 0)

    def __iter__(self):
        return self

    def __next__(self) -> Image:
        if image := self.read():
            return image

        self.release()
        raise StopIteration()

    @classmethod
    def from_uri(cls, source: str):
        return cls(source, cv2.CAP_ANY)

    @classmethod
    def from_gst(cls, pipeline: str):
        return cls(pipeline, cv2.CAP_GSTREAMER)

    @property
    def size(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def width(self) -> int:
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> int:
        return self.__capture.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self.__capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self) -> Image | None:
        ret, frame = self.__capture.read()

        if not ret:
            return None

        return Image(data=frame)

    def open(self) -> bool:
        return self.__capture.open()

    def is_opened(self) -> bool:
        return self.__capture.isOpened()

    def release(self):
        self.__capture.release()

    def seek(self, offset: int):
        self.__capture.set(cv2.CAP_PROP_POS_FRAMES, offset)

    def tell(self) -> int:
        return int(self.__capture.get(cv2.CAP_PROP_POS_FRAMES))
