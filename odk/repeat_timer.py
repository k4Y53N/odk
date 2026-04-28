from abc import ABC, abstractmethod
from collections.abc import Callable
from threading import Event, Lock, Thread
from typing import ParamSpec, TypeVar

__all__ = [
    'RepeatTimer',
]


P = ParamSpec('P')
R = TypeVar('R')


class Runnable:
    def __init__(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self):
        return self.fn(*self.args, **self.kwargs)


class RepeatTimer(Thread, ABC):
    def __init__(self, interval: int = 0, *args, **kwargs):
        """Initialize the repeat timer thread.

        Creates synchronization primitives and hook queues used by the timer
        loop and stores the wait interval between routine executions.

        Args:
            interval (int, optional): Delay in seconds between routine ticks.
                Defaults to 0.
            *args: Positional arguments forwarded to :class:`threading.Thread`.
            **kwargs: Keyword arguments forwarded to :class:`threading.Thread`.
        """
        super().__init__(*args, **kwargs)
        self.__interval = interval
        self.__event = Event()
        self.__lock = Lock()
        self.__enter_runs = list[Runnable]()
        self.__exit_runs = list[Runnable]()

    def __enter__(self):
        for enter in self.__enter_runs:
            enter.run()

        self.__enter_runs.clear()
        self.before()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        self.after()

        for exit in self.__exit_runs:
            exit.run()

        self.__exit_runs.clear()

        return False

    @property
    def lock(self) -> Lock:
        """Thread lock used to protect timer operations."""
        return self.__lock

    @abstractmethod
    def routine(self):
        """Run one timer tick.

        This method is invoked repeatedly by :meth:`run` while the timer is
        active, waiting ``interval`` seconds between invocations. Subclasses
        must implement the work to perform for each tick.
        """

    def before(self):
        """Hook called once before the timer loop starts.

        Override in subclasses to perform setup work after enter hooks are
        executed and before the first :meth:`routine` call.
        """

    def after(self):
        """Hook called once after the timer loop stops.

        Override in subclasses to perform cleanup work after :meth:`close` is
        triggered and before exit hooks are executed.
        """

    def run(self):
        """Execute the timer loop in the thread context.

        This enters the timer context once, then repeatedly calls
        :meth:`routine` while :meth:`is_running` remains true, waiting
        ``interval`` seconds between ticks.
        """
        with self:
            while self.is_active(self.__interval):
                self.routine()

    def start(self):
        with self.__lock:
            if self.is_alive() or not self.is_active():
                return

            super().start()

    def close(self):
        """Request the timer loop to stop.

        Sets the internal stop event so :meth:`is_active` returns ``False``
        and the worker loop exits on the next check.
        """
        self.__event.set()

    def is_active(self, timeout: float = 0) -> bool:
        """Check whether the timer is still active.

        Waits up to ``timeout`` seconds for the internal stop event. Returns ``True``
        if no stop signal is received, ``False`` otherwise.

        Args:
            timeout (float, optional): Seconds to wait for a stop signal.
                Defaults to 0.

        Returns:
            bool: ``True`` if still running, ``False`` if stopped.
        """
        return not self.__event.wait(timeout)

    def add_enter_hook(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register a callback to run when entering the timer context.

        The hook is executed in registration order during :meth:`__enter__`, before
        :meth:`before` is called.

        Args:
            fn (Callable[P, R]): Callback to execute on context entry.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.
        """
        self.__enter_runs.append(Runnable(fn, *args, **kwargs))

    def add_exit_hook(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register a callback to run when leaving the timer context.

        The hook is executed in registration order during :meth:`__exit__`, after
        :meth:`close` and :meth:`after` are called.

        Args:
            fn (Callable[P, R]): Callback to execute on context exit.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.
        """
        self.__exit_runs.append(Runnable(fn, *args, **kwargs))
