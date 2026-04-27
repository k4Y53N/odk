from collections.abc import Generator
from random import randint

__all__ = [
    'ColorPool',
]


class ColorPool:
    RED = (0, 0, 255)
    ORANGE = (0, 127, 255)
    YELLOW = (0, 255, 255)
    GREEN = (0, 128, 0)
    BLUE = (255, 0, 0)
    INDIGO = (255, 0, 102)
    VIOLET = (255, 0, 139)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    PINK = (203, 192, 255)
    PURPLE = (128, 0, 128)
    BROWN = (42, 42, 165)
    LIME = (0, 255, 0)

    def __init__(self, pool: tuple[tuple[int, int, int], ...]):
        self._pool = pool

    def __len__(self):
        return len(self._pool)

    def __getitem__(self, key: int) -> tuple[int, int, int]:
        return self._pool[int(key) % len(self)]

    def __iter__(self) -> Generator[tuple[int, int, int], None, None]:
        for color in self._pool:
            yield color

    @classmethod
    def random(cls, size=16) -> 'ColorPool':
        """Build a color pool with random BGR colors.

        Args:
            size (int, optional): Number of colors to generate. Defaults to 16.

        Returns:
            ColorPool: A new pool containing ``size`` random colors.
        """
        pool = tuple(
            (randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(size)
        )

        return cls(pool)

    @classmethod
    def rainbow(cls) -> 'ColorPool':
        """Build a color pool from standard rainbow colors.

        Returns:
            ColorPool: A new pool ordered as red, orange, yellow, green, blue, indigo,
                violet.
        """
        pool = (
            cls.RED,
            cls.ORANGE,
            cls.YELLOW,
            cls.GREEN,
            cls.BLUE,
            cls.INDIGO,
            cls.VIOLET,
        )

        return cls(pool)
