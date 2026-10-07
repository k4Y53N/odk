from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from threading import Event, Lock, Thread
from typing import ParamSpec, TypeVar

__all__ = [
    'Hook',
    'RepeatTimer',
]


P = ParamSpec('P')
R = TypeVar('R')


class Hook:
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
        self.__enter_hooks = list[Hook]()
        self.__exit_hooks = list[Hook]()
        self.__before_routine_hooks = list[Hook]()
        self.__after_routine_hooks = list[Hook]()

    def __enter__(self):
        self.execute_hooks(self.__enter_hooks)
        self.__enter_hooks.clear()
        self.before()

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        self.after()
        self.execute_hooks(self.__exit_hooks)
        self.__exit_hooks.clear()
        self.__before_routine_hooks.clear()
        self.__after_routine_hooks.clear()

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
                self.execute_hooks(self.__before_routine_hooks)
                self.routine()
                self.execute_hooks(self.__after_routine_hooks)

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

    def is_active(self, timeout: float | None = 0) -> bool:
        """Check whether the timer is still active.

        Waits up to ``timeout`` seconds for the internal stop event. Returns ``True``
        if no stop signal is received, ``False`` otherwise.

        Args:
            timeout (float | None, optional): Seconds to wait for a stop signal. If
                ``None``, blocks indefinitely until stopped. Defaults to 0.

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
        self.__enter_hooks.append(Hook(fn, *args, **kwargs))

    def add_exit_hook(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register a callback to run when leaving the timer context.

        The hook is executed in registration order during :meth:`__exit__`, after
        :meth:`close` and :meth:`after` are called.

        Args:
            fn (Callable[P, R]): Callback to execute on context exit.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.
        """
        self.__exit_hooks.append(Hook(fn, *args, **kwargs))

    def add_before_routine_hook(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Register a callback to run before each timer tick.

        The hook is executed in registration order immediately before
        :meth:`routine` on every active loop iteration.

        Args:
            fn (Callable[P, R]): Callback to execute before each timer tick.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.
        """
        self.__before_routine_hooks.append(Hook(fn, *args, **kwargs))

    def add_after_routine_hook(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Register a callback to run after each timer tick.

        The hook is executed in registration order immediately after
        :meth:`routine` on every active loop iteration.

        Args:
            fn (Callable[P, R]): Callback to execute after each timer tick.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.
        """
        self.__after_routine_hooks.append(Hook(fn, *args, **kwargs))

    def execute_hooks(self, hooks: Iterable[Hook]):
        """Run each hook in iteration order.

        Args:
            hooks (Iterable[Hook]): Hooks to execute.
        """
        for hook in hooks:
            hook.run()
