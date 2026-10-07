from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

from .node import Node
from .repeat_timer import Hook

__all__ = [
    'Flow',
    'Hook',
]

P = ParamSpec('P')
R = TypeVar('R')


class Flow:
    def __init__(self, *nodes: Node):
        """Create a flow by chaining nodes together in sequence.

        Each node is linked to the next using the pipe operator, forming a processing
            pipeline.

        Args:
            *nodes: Nodes to chain together in order.
        """
        prev = None

        for node in nodes:
            if prev is None:
                prev = node
                continue

            prev | node
            prev = node

        self.nodes = nodes

    def call(
        self,
        skip_standalone: bool,
        method: Callable[Concatenate[Node, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> bool:
        """Invoke a method on each node in the flow.

        Args:
            skip_standalone (bool): If True, skip nodes marked as standalone.
            method (Callable[Concatenate[Node, P], R]): The node method to call on each
                node.
            *args (P.args): Positional arguments to pass to the method.
            **kwargs (P.kwargs): Keyword arguments to pass to the method.

        Returns:
            bool: True if all invoked methods returned truthy values.
        """
        flag = True

        for node in self.nodes:
            if skip_standalone and node.standalone:
                continue

            ret = method(node, *args, **kwargs)
            flag &= bool(ret)

        return flag

    def start(self, skip_standalone: bool = True):
        """Start all nodes in the flow.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(skip_standalone, Node.start)

    def close(self, skip_standalone: bool = True):
        """Close all nodes in the flow.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(skip_standalone, Node.close)

    def join(self, timeout: float | None = None, skip_standalone: bool = True):
        """Wait for all nodes in the flow to finish.

        Args:
            timeout (float | None, optional): Maximum time in seconds to wait for each
                node. Defaults to None.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(skip_standalone, Node.join, timeout)

    def is_alive(self, skip_standalone: bool = True) -> bool:
        """Check whether all nodes in the flow are alive.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.

        Returns:
            bool: True if all checked nodes are alive.
        """
        return self.call(skip_standalone, Node.is_alive)

    def is_active(
        self,
        timeout: float | None = 0,
        skip_standalone: bool = True,
    ) -> bool:
        """Check whether all nodes in the flow are active.

        Args:
            timeout (float | None, optional): Seconds to wait for a stop signal on each
                node. If ``None``, blocks indefinitely. Defaults to 0.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.

        Returns:
            bool: True if all checked nodes are active.
        """
        return self.call(skip_standalone, Node.is_active, timeout)

    def add_enter_hook(self, hook: Hook, skip_standalone: bool = True):
        """Register an enter hook on each node in the flow.

        Args:
            hook (Hook): The hook to register, called when a node starts processing.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(
            skip_standalone,
            Node.add_enter_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_exit_hook(self, hook: Hook, skip_standalone: bool = True):
        """Register an exit hook on each node in the flow.

        Args:
            hook (Hook): The hook to register, called when a node finishes processing.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(
            skip_standalone,
            Node.add_exit_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_before_routine_hook(self, hook: Hook, skip_standalone: bool = True):
        """Register a hook that runs before each node routine.

        Args:
            hook (Hook): The hook to register before a node routine executes.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(
            skip_standalone,
            Node.add_before_routine_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_after_routine_hook(self, hook: Hook, skip_standalone: bool = True):
        """Register a hook that runs after each node routine.

        Args:
            hook (Hook): The hook to register after a node routine executes.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(
            skip_standalone,
            Node.add_after_routine_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_self_close_hook(self, skip_standalone: bool = True):
        """Register an exit hook that closes the entire flow when any node exits.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.add_exit_hook(Hook(self.close, skip_standalone), skip_standalone)
