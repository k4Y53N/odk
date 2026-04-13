from dataclasses import dataclass, field
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

__all__ = [
    'Image',
]

EXTENSION = Literal['.jpg', '.jpeg', '.jpe', '.png', '.webp']


def get_image_ext_quality(extension: EXTENSION, quality: float) -> list[int]:
    """Return OpenCV encoding parameters for the given image format and quality.

    Args:
        extension (EXTENSION): Image format extension including the leading dot.
        quality (float): Normalized quality value in the range [0, 1].

    Raises:
        ValueError: If *quality* is not between 0 and 1.

    Returns:
        list[int]: A list of OpenCV ``imwrite`` parameter flag and value pairs.
    """
    if not 0 <= quality <= 1:
        raise ValueError(f'quality must be between 0 and 1, got {quality}')

    if extension in {'.jpg', '.jpeg', '.jpe'}:
        return [cv2.IMWRITE_JPEG_QUALITY, round(quality * 100)]

    if extension == '.png':
        return [cv2.IMWRITE_PNG_COMPRESSION, round((1 - quality) * 9)]

    if extension == '.webp':
        return [cv2.IMWRITE_WEBP_QUALITY, round(quality * 100)]

    return []


@dataclass(slots=True)
class Image:
    """Wrapper around a NumPy image array with convenience I/O and display methods."""

    data: NDArray[np.uint8]
    timestamp: datetime = field(default_factory=lambda: datetime.now().astimezone())

    @property
    def width(self) -> int:
        """Return the image width in pixels."""
        return self.data.shape[1]

    @property
    def height(self) -> int:
        """Return the image height in pixels."""
        return self.data.shape[0]

    @property
    def size(self) -> tuple[int, int]:
        """Return the image size as ``(height, width)``."""
        return (self.height, self.width)

    @classmethod
    def from_file(cls, path: str | PathLike, flags: int = cv2.IMREAD_COLOR) -> 'Image':
        """Load an image from a file on disk.

        Args:
            path (str | PathLike): Path to the image file.
            flags (int, optional): OpenCV imread flags. Defaults to cv2.IMREAD_COLOR.

        Raises:
            FileNotFoundError: If the file does not exist or cannot be read.

        Returns:
            Image: A new ``Image`` instance with the loaded pixel data.
        """
        data = cv2.imread(path, flags)

        if data is not None:
            return Image(data)

        raise FileNotFoundError(f'Image path: {path} not avaliable')

    @classmethod
    def decode(
        cls,
        buffer: NDArray[np.uint8],
        flags: int = cv2.IMREAD_COLOR,
    ) -> 'Image':
        """Decode an image from an in-memory byte buffer.

        Args:
            buffer (NDArray[np.uint8]): Encoded image bytes as a NumPy array.
            flags (int, optional): OpenCV imread flags. Defaults to cv2.IMREAD_COLOR.

        Raises:
            RuntimeError: If the buffer cannot be decoded.

        Returns:
            Image: A new ``Image`` instance with the decoded pixel data.
        """
        buffer = cv2.imdecode(buffer, flags=flags)

        if buffer is not None:
            return cls(data=buffer)

        raise RuntimeError('Decode from buffer error')

    @classmethod
    def decode_bytes(
        cls,
        byte_buffer: bytes,
        flags: int = cv2.IMREAD_COLOR,
    ) -> 'Image':
        """Decode an image from raw ``bytes``.

        Args:
            byte_buffer (bytes): Encoded image data as a Python bytes object.
            flags (int, optional): OpenCV imread flags. Defaults to cv2.IMREAD_COLOR.

        Returns:
            Image: A new ``Image`` instance with the decoded pixel data.
        """
        buffer = np.frombuffer(byte_buffer, dtype=np.uint8)
        return cls.decode(buffer, flags)

    def copy(self) -> 'Image':
        """Return a deep copy of the image data preserving the timestamp."""
        return Image(self.data.copy(), timestamp=self.timestamp)

    def resize(self, width: int, height: int, interpolation: int = cv2.INTER_LINEAR):
        """Resize the image in-place to the given dimensions.

        Args:
            width (int): Target width in pixels.
            height (int): Target height in pixels.
            interpolation (int, optional): OpenCV interpolation flag. Defaults to
                cv2.INTER_LINEAR.
        """
        self.data = cv2.resize(self.data, (width, height), interpolation=interpolation)

    def encode(
        self,
        extension: EXTENSION = '.jpg',
        quality: float = 1,
    ) -> NDArray[np.uint8]:
        """Encode the image into an OpenCV byte buffer.

        Args:
            extension (EXTENSION, optional): Output format extension including the
                leading dot. Defaults to '.jpg'.
            quality (float, optional): Normalized quality value in the range [0, 1].
                Defaults to 1.

        Raises:
            RuntimeError: If encoding fails.

        Returns:
            NDArray[np.uint8]: Encoded image bytes stored in a ``uint8`` NumPy array.
        """
        quality_params = get_image_ext_quality(extension, quality)
        is_success, buf = cv2.imencode(extension, self.data, quality_params)

        if is_success:
            return buf

        raise RuntimeError(f'Encode image to {extension} failed')

    def encode_bytes(
        self,
        extension: EXTENSION = '.jpg',
        quality: float = 1,
    ) -> bytes:
        """Encode the image and return the result as Python ``bytes``.

        Args:
            extension (EXTENSION, optional): Output format extension including the
                leading dot. Defaults to '.jpg'.
            quality (float, optional): Normalized quality value in the range [0, 1].
                Defaults to 1.

        Returns:
            bytes: The encoded image data.
        """
        buf = self.encode(extension, quality)
        return buf.tobytes()

    def save(self, path: str | PathLike, quality: float = 1):
        """Save the image to a file on disk.

        Args:
            path (str | PathLike): Destination file path. The format is inferred from
                the file extension.
            quality (float, optional): Normalized quality value in the range [0, 1].
                Defaults to 1.

        Raises:
            OSError: If the image could not be written to *path*.
        """
        suffix = Path(path).suffix
        quality_params = get_image_ext_quality(suffix, quality)
        is_saved = cv2.imwrite(path, self.data, quality_params)

        if not is_saved:
            raise OSError(f'Failed to save image to {path}')

    def show(self, delay: int = 0, window_name: str = 'image') -> int:
        """Display the image in an OpenCV window and wait for a key press.

        Args:
            delay (int, optional): Milliseconds to wait for a key press. 0 waits
                indefinitely. Defaults to 0.
            window_name (str, optional): Title of the display window. Defaults to
                'image'.

        Returns:
            int: The code of the key pressed, or -1 if no key was pressed before the
                timeout.
        """
        cv2.imshow(window_name, self.data)
        return cv2.waitKey(delay=delay)

    def interrupt_show(
        self,
        delay: int = 0,
        interrupt_keys: Iterable[str] = 'qQ',
        window_name: str | None = None,
    ):
        """Display the image and raise ``KeyboardInterrupt`` if an interrupt key is
        pressed.

        Args:
            delay (int, optional): Milliseconds to wait for a key press. 0 waits
                indefinitely. Defaults to 0.
            interrupt_keys (Iterable[str], optional): Characters that trigger an
                interrupt. Defaults to 'qQ'.
            window_name (str | None, optional): Title of the display window.
                Auto-generated from *interrupt_keys* when ``None``. Defaults to None.

        Raises:
            KeyboardInterrupt: If the pressed key is in *interrupt_keys* (or any key
                when *interrupt_keys* is empty).

        Returns:
            int: The key code if it was not an interrupt key.
        """
        interrupt_keys = tuple(interrupt_keys)

        if window_name is None:
            window_name = 'Interrupt by key: ' + ' '.join(interrupt_keys)

        key = self.show(delay=delay, window_name=window_name)

        if (key != -1 and len(interrupt_keys) == 0) or key in interrupt_keys:
            raise KeyboardInterrupt(f'Interrupt by key {chr(key)}')

        return key

    def draw_text(
        self,
        text: str,
        left: float,
        top: float,
        color: tuple[int, int, int],
        font_scale=1,
        thickness=1,
        font_face=cv2.FONT_HERSHEY_DUPLEX,
        background=False,
    ) -> tuple[int, int]:
        """Draw multi-line text onto the image.

        Renders *text* at the given pixel position, splitting on newline characters so
        each line is drawn below the previous one.  When *background* is enabled, a
        filled rectangle using the inverted *color* is drawn behind each line for
        contrast.

        Args:
            text (str): The text string to render. Newline characters split the text
                into multiple lines.
            left (float): X-coordinate of the left edge of the text, in pixels.
            top (float): Y-coordinate of the top edge of the first line, in pixels.
            color (tuple[int, int, int]): BGR text color. When *background* is ``True``
                this becomes the background color and the text is drawn in the inverted
                color.
            font_scale (int, optional): Font size multiplier. Defaults to 1.
            thickness (int, optional): Thickness of the text strokes in pixels.
                Defaults to 1.
            font_face (int, optional): OpenCV font identifier. Defaults to
                cv2.FONT_HERSHEY_DUPLEX.
            background (bool, optional): If ``True``, draw a filled rectangle behind
                each line using the inverted *color*. Defaults to False.

        Returns:
            tuple[int, int]: The ``(x, y)`` pixel coordinates of the bottom-right
                anchor after the last rendered line.
        """
        left, top = round(left), round(top)
        lines = text.split('\n')
        background_color = (255 - color[0], 255 - color[1], 255 - color[2])
        anchor_x = left
        anchor_y = top

        if background:
            color, background_color = background_color, color

        for line in lines:
            (text_width, text_height), baseline = cv2.getTextSize(
                text=line,
                fontFace=font_face,
                fontScale=font_scale,
                thickness=thickness,
            )
            total_width = left + text_width
            total_height = top + text_height + baseline
            anchor_x = left + text_width
            anchor_y = top + text_height

            if background:
                cv2.rectangle(
                    self.data,
                    (left, top),
                    (total_width, total_height),
                    color=background_color,
                    thickness=-1,
                )

            cv2.putText(
                self.data,
                text=line,
                org=(left, anchor_y),
                fontFace=font_face,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
            )
            top = total_height

        return anchor_x, top

    def draw_bbox(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        color: tuple[int, int, int],
        thickness: int = 2,
    ):
        """Draw a bounding box rectangle on the image.

        Args:
            left (float): X-coordinate of the left edge, in pixels.
            top (float): Y-coordinate of the top edge, in pixels.
            right (float): X-coordinate of the right edge, in pixels.
            bottom (float): Y-coordinate of the bottom edge, in pixels.
            color (tuple[int, int, int]): BGR color of the rectangle outline.
            thickness (int, optional): Line thickness in pixels. Defaults to 2.
        """
        left, top, right, bottom = np.round([left, top, right, bottom]).astype(np.int32)
        cv2.rectangle(
            self.data,
            pt1=(left, top),
            pt2=(right, bottom),
            color=color,
            thickness=thickness,
        )

    def draw_bboxes(
        self,
        bboxes: NDArray[np.int_] | NDArray[np.float32],
        color: tuple[int, int, int],
        thickness=2,
    ):
        """Draw multiple bounding box rectangles on the image.

        Args:
            bboxes (NDArray[np.int_] | NDArray[np.float32]): Array of shape ``(N, 4)``
                where each row is ``[left, top, right, bottom]``.
            color (tuple[int, int, int]): BGR color of the rectangle outlines.
            thickness (int, optional): Line thickness in pixels. Defaults to 2.
        """
        for bbox in bboxes:
            self.draw_bbox(*bbox, color=color, thickness=thickness)

    def draw_line(
        self,
        points: NDArray[np.int_] | NDArray[np.float32],
        color: tuple[int, int, int],
        is_closed: bool = False,
        thickness: int = 2,
    ):
        """Draw a polyline through the given points on the image.

        Args:
            points (ArrayLike[np.int_] | ArrayLike[np.float32]): Array of shape
                ``(N, 2)`` containing ``(x, y)`` vertex coordinates.
            color (tuple[int, int, int]): BGR color of the line.
            is_closed (bool, optional): If ``True``, connect the last point back to the
                first. Defaults to False.
            thickness (int, optional): Line thickness in pixels. Defaults to 2.
        """
        points = np.round(points).astype(np.int32)
        cv2.polylines(
            self.data,
            pts=[points],
            color=color,
            isClosed=is_closed,
            thickness=thickness,
        )

    def draw_polygon(
        self,
        points: NDArray[np.int_] | NDArray[np.float32],
        color: tuple[int, int, int],
        alpha: float = 1,
    ):
        """Draw a filled polygon on the image.

        When *alpha* is ``1`` the polygon is drawn opaquely. Values between ``0`` and
        ``1`` blend the polygon with the existing image. An *alpha* of ``0`` is a
        no-op.

        Args:
            points (NDArray[np.int_] | NDArray[np.float32]): Array of shape ``(N, 2)``
                containing ``(x, y)`` vertex coordinates.
            color (tuple[int, int, int]): BGR fill color.
            alpha (float, optional): Opacity in the range [0, 1]. Defaults to 1.
        """
        alpha = np.clip(alpha, 0, 1)

        if not alpha:
            return

        points = np.round(points).astype(np.int32)

        if alpha == 1:
            cv2.fillPoly(self.data, pts=[points], color=color)
            return

        overlay = self.data.copy()
        cv2.fillPoly(overlay, pts=[points], color=color)
        cv2.addWeighted(overlay, alpha, self.data, 1 - alpha, 0, dst=self.data)
