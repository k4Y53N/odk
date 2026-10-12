from threading import Lock, Thread

from .lifecycle import Lifecycle

__all__ = ['RepeatTimer']


class RepeatTimer(Lifecycle, Thread):
    def __init__(self, interval: float = 0, *args, **kwargs):
        Lifecycle.__init__(self)
        Thread.__init__(self, *args, **kwargs)
        self.__interval = interval
        self.__lock = Lock()

    @property
    def lock(self) -> Lock:
        """Thread lock used to protect timer operations."""
        return self.__lock

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
