from collections.abc import Callable, Generator
from datetime import datetime
from functools import wraps
from random import randint
from time import perf_counter
from typing import ParamSpec, TypeVar

__all__ = [
    'now',
    'timeit',
    'ColorPool',
]


def now() -> datetime:
    """Return the current local time as a timezone-aware datetime.

    Returns:
        datetime: Current timestamp in the system local timezone.
    """
    return datetime.now().astimezone()


P = ParamSpec('P')
R = TypeVar('R')


def timeit(func: Callable[P, R]) -> Callable[P, R]:
    """Measure and print the runtime of the decorated function.

    Args:
        func (Callable[P, R]): Function to wrap.

    Returns:
        Callable[P, R]: Wrapped function that prints elapsed time in seconds.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = perf_counter()
        ret = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f'{func.__qualname__} executed in {elapsed:.6f}s')

        return ret

    return wrapper


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
