from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

from .lifecycle import Hook
from .node import NodeWorker

__all__ = [
    'Flow',
]

P = ParamSpec('P')
R = TypeVar('R')


class BaseFlow(ABC):
    @abstractmethod
    def call(
        self,
        hook: Hook[Concatenate[NodeWorker, P], R],
        skip_standalone: bool,
    ) -> bool:
        """Invoke a hook on each selected node.

        Implementations call ``hook`` with each node as its first argument and
        combine the hook results into a single boolean value.

        Args:
            hook (Hook[Concatenate[NodeWorker, P], R]): Hook wrapping a
                NodeWorker method to invoke on each selected node.
            skip_standalone (bool): Whether to exclude nodes marked as standalone.

        Returns:
            bool: True when every invoked hook result is truthy.
        """

    @abstractmethod
    def close(self, skip_standalone: bool = True):
        """Close selected nodes in the flow."""


class FlowHooks:
    def __init__(self, flow: BaseFlow, skip_standalone: bool):
        self.__flow = flow
        self.__skip_standalone = skip_standalone

    def add_enter(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register an enter hook on each selected node."""
        self.__flow.call(
            Hook(NodeWorker.add_enter_hook, fn, *args, **kwargs),
            self.__skip_standalone,
        )
        return self

    def add_exit(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register an exit hook on each selected node."""
        self.__flow.call(
            Hook(NodeWorker.add_exit_hook, fn, *args, **kwargs),
            self.__skip_standalone,
        )
        return self

    def add_before_routine(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Register a hook before each selected node routine."""
        self.__flow.call(
            Hook(NodeWorker.add_before_routine_hook, fn, *args, **kwargs),
            self.__skip_standalone,
        )
        return self

    def add_after_routine(
        self,
        fn: Callable[P, R],
        *args: P.args,
        **kwargs: P.kwargs,
    ):
        """Register a hook after each selected node routine."""
        self.__flow.call(
            Hook(NodeWorker.add_after_routine_hook, fn, *args, **kwargs),
            self.__skip_standalone,
        )
        return self

    def add_self_close(self):
        """Close selected nodes when any selected node exits."""
        return self.add_exit(self.__flow.close, self.__skip_standalone)


class Flow(BaseFlow):
    def __init__(self, *nodes: NodeWorker):
        """Create a flow by chaining nodes together in sequence.

        Each node is linked to the next using the pipe operator, forming a processing
            pipeline.

        Args:
            *nodes: ThreadNodes to chain together in order.
        """
        prev = None

        for node in nodes:
            if prev is None:
                prev = node
                continue

            prev | node
            prev = node

        self.nodes = nodes

    def call(self, hook, skip_standalone):
        flag = True

        for node in self.nodes:
            if skip_standalone and node.standalone:
                continue

            ret = hook.fn(node, *hook.args, **hook.kwargs)
            flag &= bool(ret)

        return flag

    def start(self, skip_standalone: bool = True):
        """Start all nodes in the flow.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(Hook(NodeWorker.start), skip_standalone)

    def close(self, skip_standalone: bool = True):
        """Close all nodes in the flow.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(Hook(NodeWorker.close), skip_standalone)

    def join(self, timeout: float | None = None, skip_standalone: bool = True):
        """Wait for all nodes in the flow to finish.

        Args:
            timeout (float | None, optional): Maximum time in seconds to wait for each
                node. Defaults to None.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(Hook(NodeWorker.join, timeout), skip_standalone)

    def is_alive(self, skip_standalone: bool = True) -> bool:
        """Check whether all nodes in the flow are alive.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.

        Returns:
            bool: True if all checked nodes are alive.
        """
        return self.call(Hook(NodeWorker.is_alive), skip_standalone)

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
        return self.call(Hook(NodeWorker.is_active, timeout), skip_standalone)

    def hook(self, skip_standalone: bool = True) -> FlowHooks:
        """Create a lifecycle hook helper.

        Args:
            skip_standalone (bool, optional): Skip standalone nodes. Defaults to True.

        Returns:
            FlowHooks: Hook helper.
        """
        return FlowHooks(self, skip_standalone)
