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
            interpolation (int, optional): OpenCV interpolation flag. Defaults to cv2.INTER_LINEAR.
        """
        self.data = cv2.resize(self.data, (width, height), interpolation=interpolation)

    def encode(
        self,
        extension: EXTENSION = '.jpg',
        quality: float = 1,
    ) -> NDArray[np.uint8]:
        """Encode the image into an OpenCV byte buffer.

        Args:
            extension (EXTENSION, optional): Output format extension including the leading dot. Defaults to '.jpg'.
            quality (float, optional): Normalized quality value in the range [0, 1]. Defaults to 1.

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
            extension (EXTENSION, optional): Output format extension including the leading dot. Defaults to '.jpg'.
            quality (float, optional): Normalized quality value in the range [0, 1]. Defaults to 1.

        Returns:
            bytes: The encoded image data.
        """
        buf = self.encode(extension, quality)
        return buf.tobytes()

    def save(self, path: str | PathLike, quality: float = 1):
        """Save the image to a file on disk.

        Args:
            path (str | PathLike): Destination file path. The format is inferred from the file extension.
            quality (float, optional): Normalized quality value in the range [0, 1]. Defaults to 1.

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
            delay (int, optional): Milliseconds to wait for a key press. 0 waits indefinitely. Defaults to 0.
            window_name (str, optional): Title of the display window. Defaults to 'image'.

        Returns:
            int: The code of the key pressed, or -1 if no key was pressed before the timeout.
        """
        cv2.imshow(window_name, self.data)
        return cv2.waitKey(delay=delay)

    def interrupt_show(
        self,
        delay: int = 0,
        interrupt_keys: Iterable[str] = 'qQ',
        window_name: str | None = None,
    ):
        """Display the image and raise ``KeyboardInterrupt`` if an interrupt key is pressed.

        Args:
            delay (int, optional): Milliseconds to wait for a key press. 0 waits indefinitely. Defaults to 0.
            interrupt_keys (Iterable[str], optional): Characters that trigger an interrupt. Defaults to 'qQ'.
            window_name (str | None, optional): Title of the display window. Auto-generated from *interrupt_keys* when ``None``. Defaults to None.

        Raises:
            KeyboardInterrupt: If the pressed key is in *interrupt_keys* (or any key when *interrupt_keys* is empty).

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
