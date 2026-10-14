from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

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
        skip_standalone: bool,
        method: Callable[Concatenate[NodeWorker, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> bool:
        """Invoke a method on each node in the flow.

        Args:
            skip_standalone (bool): If True, skip nodes marked as standalone.
            method (Callable[Concatenate[ThreadNode, P], R]): The node method to call on each
                node.
            *args (P.args): Positional arguments to pass to the method.
            **kwargs (P.kwargs): Keyword arguments to pass to the method.

        Returns:
            bool: True if all invoked methods returned truthy values.
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
            self.__skip_standalone,
            NodeWorker.add_enter_hook,
            fn,
            *args,
            **kwargs,
        )
        return self

    def add_exit(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        """Register an exit hook on each selected node."""
        self.__flow.call(
            self.__skip_standalone,
            NodeWorker.add_exit_hook,
            fn,
            *args,
            **kwargs,
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
            self.__skip_standalone,
            NodeWorker.add_before_routine_hook,
            fn,
            *args,
            **kwargs,
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
            self.__skip_standalone,
            NodeWorker.add_after_routine_hook,
            fn,
            *args,
            **kwargs,
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

    def call(
        self,
        skip_standalone: bool,
        method: Callable[Concatenate[NodeWorker, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> bool:
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
        self.call(skip_standalone, NodeWorker.start)

    def close(self, skip_standalone: bool = True):
        """Close all nodes in the flow.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(skip_standalone, NodeWorker.close)

    def join(self, timeout: float | None = None, skip_standalone: bool = True):
        """Wait for all nodes in the flow to finish.

        Args:
            timeout (float | None, optional): Maximum time in seconds to wait for each
                node. Defaults to None.
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.
        """
        self.call(skip_standalone, NodeWorker.join, timeout)

    def is_alive(self, skip_standalone: bool = True) -> bool:
        """Check whether all nodes in the flow are alive.

        Args:
            skip_standalone (bool, optional): If True, skip nodes marked as standalone.
                Defaults to True.

        Returns:
            bool: True if all checked nodes are alive.
        """
        return self.call(skip_standalone, NodeWorker.is_alive)

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
        return self.call(skip_standalone, NodeWorker.is_active, timeout)

    def hook(self, skip_standalone: bool = True) -> FlowHooks:
        """Create a lifecycle hook helper.

        Args:
            skip_standalone (bool, optional): Skip standalone nodes. Defaults to True.

        Returns:
            FlowHooks: Hook helper.
        """
        return FlowHooks(self, skip_standalone)
