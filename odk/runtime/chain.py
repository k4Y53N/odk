from contextlib import ExitStack
from typing import Any, Generic, TypeVar

from .node import Node
from .repeat_timer import RepeatTimer

IN = TypeVar('IN')
OUT = TypeVar('OUT')


__all__ = [
    'Chain',
]


class Chain(RepeatTimer, Generic[IN, OUT]):
    def __init__(
        self,
        nodes: list['Node[Any, Any]'],
        interval: float = 0,
        *args,
        **kwargs,
    ):
        """Create a chain that processes items through nodes in order.

        Args:
            nodes (list[Node[Any, Any]]): Nodes to execute in sequence.
            interval (float, optional): Delay between processing iterations in seconds.
                Defaults to 0.

        Raises:
            ValueError: If ``nodes`` is empty.
        """
        if not nodes:
            raise ValueError('Chain requires at least one node.')

        super().__init__(interval, *args, **kwargs)
        self.__nodes = nodes
        self.__node_contexts: ExitStack | None = None

        for node in nodes:
            node.bind_event(self.event)

    def __enter__(self):
        super().__enter__()
        self.__node_contexts = ExitStack()

        try:
            for node in self.__nodes:
                self.__node_contexts.enter_context(node)
        except BaseException:
            self.__node_contexts.close()
            self.__node_contexts = None
            super().__exit__(None, None, None)
            raise

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        suppressed = False

        if self.__node_contexts is not None:
            suppressed = self.__node_contexts.__exit__(exc_type, exc_val, exc_tb)
            self.__node_contexts = None

        return super().__exit__(exc_type, exc_val, exc_tb) or suppressed

    def routine(self):
        self.process(None)

    def process(self, item: IN | None) -> OUT | None:
        """Process an item through each node in sequence.

        Args:
            item (IN | None): Initial item passed to the first node.

        Returns:
            OUT | None: Output returned by the final node.
        """
        for node in self.__nodes:
            node.execute_before_routine_hooks()
            item = node.process(item)
            node.execute_after_routine_hooks()

        return item
