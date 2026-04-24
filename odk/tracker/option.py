from dataclasses import dataclass
from typing import Literal

from .tracker import Tracker

__all__ = [
    'TrackOption',
]


@dataclass(slots=True, kw_only=True)
class TrackOption:
    timeout: int = 10
    tracker_type: Literal['SORT'] = 'SORT'
    sort_threshold: float = 0.3

    def create(self) -> Tracker:
        """Create and return a Tracker instance based on the current TrackOption
        settings.

        Returns:
            Tracker: An instance of the selected tracker type (currently only 'SORT' is supported).

        Raises:
            NotImplementedError: If the specified tracker_type is not supported.
        """
        if self.tracker_type == 'SORT':
            from .sort import SortTracker

            return SortTracker(
                timeout=self.timeout,
                threshold=self.sort_threshold,
            )

        raise NotImplementedError(f'tracker_type not suppored: {self.tracker_type}')
