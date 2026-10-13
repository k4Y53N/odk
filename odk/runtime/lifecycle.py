from collections.abc import Callable, Iterable
from typing import Generic, ParamSpec, TypeVar

__all__ = [
    'Hook',
    'Lifecycle',
]


P = ParamSpec('P')
R = TypeVar('R')
Self = TypeVar('Self', bound='Lifecycle')


class Hook(Generic[P, R]):
    def __init__(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

    def run(self) -> R:
        return self.fn(*self.args, **self.kwargs)


class Lifecycle:
    def __init__(self):
        self.__enter_hooks = list[Hook]()
        self.__exit_hooks = list[Hook]()
        self.__before_routine_hooks = list[Hook]()
        self.__after_routine_hooks = list[Hook]()

    def __enter__(self: Self) -> Self:
        self.execute_enter_hooks()
        self.__enter_hooks.clear()
        self.before()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.after()
        self.execute_exit_hooks()
        self.__exit_hooks.clear()
        self.__before_routine_hooks.clear()
        self.__after_routine_hooks.clear()

        return False

    def routine(self):
        """Perform one unit of work while the lifecycle is active.

        Override this method to implement the work for one iteration. It runs
        after before-routine hooks and before after-routine hooks while the
        lifecycle remains active.
        """

    def before(self):
        """Perform setup before routine processing begins.

        Override this method to initialize resources after enter hooks run and
        before the first :meth:`routine` call.
        """

    def after(self):
        """Perform cleanup when the lifecycle is exited.

        Override this method to release resources after routine processing and
        before exit hooks run.
        """

    def add_enter_hook(
        self: Self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Register a callback to run when the lifecycle is entered.

        The callback runs in registration order before :meth:`before`. The
        returned instance allows chained hook registration.

        Args:
            fn (Callable[P, R]): Callback to execute on context entry.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.

        Returns:
            Self: self.
        """
        self.__enter_hooks.append(Hook(fn, *args, **kwargs))
        return self

    def add_exit_hook(
        self: Self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Register a callback to run when the lifecycle is exited.

        The callback runs in registration order after :meth:`after`. The
        returned instance allows chained hook registration.

        Args:
            fn (Callable[P, R]): Callback to execute on context exit.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.

        Returns:
            Self: self.
        """
        self.__exit_hooks.append(Hook(fn, *args, **kwargs))
        return self

    def add_before_routine_hook(
        self: Self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Register a callback to run before each routine invocation.

        Callbacks run in registration order immediately before :meth:`routine`
        during each active iteration. The returned instance allows chained hook
        registration.

        Args:
            fn (Callable[P, R]): Callback to execute before each routine call.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.

        Returns:
            Self: self.
        """
        self.__before_routine_hooks.append(Hook(fn, *args, **kwargs))
        return self

    def add_after_routine_hook(
        self: Self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Register a callback to run after each routine invocation.

        Callbacks run in registration order immediately after :meth:`routine`
        during each active iteration. The returned instance allows chained hook
        registration.

        Args:
            fn (Callable[P, R]): Callback to execute after each routine call.
            *args (P.args): Positional arguments passed to ``fn``.
            **kwargs (P.kwargs): Keyword arguments passed to ``fn``.

        Returns:
            Self: self.
        """
        self.__after_routine_hooks.append(Hook(fn, *args, **kwargs))
        return self

    def execute_hooks(self, hooks: Iterable[Hook]):
        """Run each hook in iteration order.

        Args:
            hooks (Iterable[Hook]): Hooks to execute.
        """
        for hook in hooks:
            hook.run()

    def execute_enter_hooks(self):
        """Run registered entry hooks in registration order.

        Called when entering the lifecycle, before :meth:`before`.
        """
        self.execute_hooks(self.__enter_hooks)

    def execute_exit_hooks(self):
        """Run registered exit hooks in registration order.

        Called when exiting the lifecycle, after :meth:`after`.
        """
        self.execute_hooks(self.__exit_hooks)

    def execute_before_routine_hooks(self):
        """Run registered before-routine hooks in registration order.

        Called immediately before :meth:`routine` on each active iteration.
        """
        self.execute_hooks(self.__before_routine_hooks)

    def execute_after_routine_hooks(self):
        """Run registered after-routine hooks in registration order.

        Called immediately after :meth:`routine` on each active iteration.
        """
        self.execute_hooks(self.__after_routine_hooks)
