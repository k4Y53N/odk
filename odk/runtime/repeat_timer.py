from threading import Event, Lock, Thread

from .lifecycle import Lifecycle

__all__ = ['RepeatTimer']


class RepeatTimer(Lifecycle, Thread):
    def __init__(self, interval: float = 0, *args, **kwargs):
        Lifecycle.__init__(self)
        Thread.__init__(self, *args, **kwargs)
        self.__interval = interval
        self.__lock = Lock()
        self.__event = Event()

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return super().__exit__(exc_type, exc_val, exc_tb)

    @property
    def lock(self) -> Lock:
        """Thread lock used to protect timer operations."""
        return self.__lock

    @property
    def event(self) -> Event:
        return self.__event

    def run(self):
        with self:
            while self.is_active(self.__interval):
                self.execute_before_routine_hooks()
                self.routine()
                self.execute_after_routine_hooks()

    def start(self):
        with self.lock:
            if self.is_alive() or not self.is_active():
                return

            super().start()

    def close(self):
        """Request that the timer stop.

        Sets the internal stop event so :meth:`is_active` returns ``False``.
        The active routine loop ends on its next activity check.
        """
        self.__event.set()

    def is_active(self, timeout: float | None = 0) -> bool:
        """Check whether the timer remains active.

        Waits up to ``timeout`` seconds for the internal stop event. Returns
        ``True`` when no stop signal is received, otherwise ``False``.

        Args:
            timeout (float | None, optional): Seconds to wait for a stop signal. If
                ``None``, blocks indefinitely until stopped. Defaults to 0.

        Returns:
            bool: ``True`` if still running, ``False`` if stopped.
        """
        return not self.__event.wait(timeout)
