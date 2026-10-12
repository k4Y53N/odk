from collections.abc import Callable, Iterable
from threading import Event
from typing import Generic, ParamSpec, TypeVar

__all__ = [
    'Hook',
    'Lifecycle',
]


P = ParamSpec('P')
R = TypeVar('R')


class Hook(Generic[P, R]):
    def __init__(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> R:
        return self.fn(*self.args, **self.kwargs)


class Lifecycle:
    def __init__(self):
        self.__event = Event()
        self.__enter_hooks = list[Hook]()
        self.__exit_hooks = list[Hook]()
        self.__before_routine_hooks = list[Hook]()
        self.__after_routine_hooks = list[Hook]()

    def __enter__(self):
        self.execute_enter_hooks()
        self.__enter_hooks.clear()
        self.before()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        self.after()
        self.execute_exit_hooks()
        self.__exit_hooks.clear()
        self.__before_routine_hooks.clear()
        self.__after_routine_hooks.clear()

        return False

    def routine(self):
        """Perform one unit of work for an active timer tick.

        Override in subclasses to provide the work that runs after before-tick
        hooks and before after-tick hooks on each timer iteration.
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

    def execute_enter_hooks(self):
        self.execute_hooks(self.__enter_hooks)

    def execute_exit_hooks(self):
        self.execute_hooks(self.__exit_hooks)

    def execute_before_routine_hooks(self):
        self.execute_hooks(self.__before_routine_hooks)

    def execute_after_routine_hooks(self):
        self.execute_hooks(self.__after_routine_hooks)
