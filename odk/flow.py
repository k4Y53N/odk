from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

from .node import Node

__all__ = [
    'Flow',
    'Hook',
]

P = ParamSpec('P')
R = TypeVar('R')


class Hook:
    def __init__(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Create a hook with a callable and its arguments.

        Args:
            fn (Callable[P, R]): The function to invoke when the hook is triggered.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        """
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


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
        skip_keep_alive: bool,
        method: Callable[Concatenate[Node, ...]],
        *args,
        **kwargs,
    ) -> bool:
        """Invoke a method on each node in the flow.

        Args:
            skip_keep_alive (bool): If True, skip nodes marked as keep_alive.
            method (Callable[Concatenate[Node, ...]]): The node method to call on each
                node.
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            bool: True if all invoked methods returned truthy values.
        """
        flag = True

        for node in self.nodes:
            if skip_keep_alive and node.keep_alive:
                continue

            ret = method(node, *args, **kwargs)
            flag &= bool(ret)

        return flag

    def start(self, skip_keep_alive: bool = True):
        """Start all nodes in the flow.

        Args:
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
        """
        self.call(skip_keep_alive, Node.start)

    def close(self, skip_keep_alive: bool = True):
        """Close all nodes in the flow.

        Args:
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
        """
        self.call(skip_keep_alive, Node.close)

    def join(self, skip_keep_alive: bool = True, timeout: float | None = None):
        """Wait for all nodes in the flow to finish.

        Args:
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
            timeout (float | None, optional): Maximum time in seconds to wait for each
                node. Defaults to None.
        """
        self.call(skip_keep_alive, Node.join, timeout)

    def is_active(self, skip_keep_alive: bool = True) -> bool:
        """Check whether all nodes in the flow are active.

        Args:
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.

        Returns:
            bool: True if all checked nodes are active.
        """
        return self.call(skip_keep_alive, Node.is_active)

    def add_enter_hook(self, hook: Hook, skip_keep_alive: bool = True):
        """Register an enter hook on each node in the flow.

        Args:
            hook (Hook): The hook to register, called when a node starts processing.
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
        """
        self.call(
            skip_keep_alive,
            Node.add_enter_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_exit_hook(self, hook: Hook, skip_keep_alive: bool = True):
        """Register an exit hook on each node in the flow.

        Args:
            hook (Hook): The hook to register, called when a node finishes processing.
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
        """
        self.call(
            skip_keep_alive,
            Node.add_exit_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_self_close_hook(self, skip_keep_alive: bool = True):
        """Register an exit hook that closes the entire flow when any node exits.

        Args:
            skip_keep_alive (bool, optional): If True, skip nodes marked as keep_alive.
                Defaults to True.
        """
        self.add_exit_hook(Hook(self.close, skip_keep_alive), skip_keep_alive)
