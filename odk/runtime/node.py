import logging as log
from abc import ABC, abstractmethod
from queue import Empty, Full, Queue
from threading import Event
from typing import Any, Generic, TypeVar

from .lifecycle import Lifecycle
from .repeat_timer import RepeatTimer

__all__ = [
    'Node',
    'NodeWorker',
]


IN = TypeVar('IN')
OUT = TypeVar('OUT')


class BaseNode(Lifecycle, Generic[IN, OUT], ABC):
    def __init__(self):
        super().__init__()
        self.__event = Event()

    @abstractmethod
    def process(self, item: IN | None) -> OUT | None:
        """Process one input item.

        Args:
            item (IN | None): Input from the upstream node, or ``None`` when no
                input item is available.

        Returns:
            OUT | None: The item to send downstream, or ``None`` to emit nothing.
        """

    def bind_event(self, event: Event):
        """Bind the runtime stop event used by :meth:`close` and :meth:`is_active`.

        Args:
            event (Event): Stop event owned by the runtime that executes this node.
        """
        self.__event = event

    def close(self):
        """Set the node to inactive."""
        self.__event.set()

    def is_active(self, timeout: float | None):
        """Check whether the node is active.

        Args:
            timeout (float | None): Seconds to wait for the node to close. ``None``
                waits indefinitely.

        Returns:
            bool: ``True`` if the node remains active, otherwise ``False``.
        """
        return not self.__event.wait(timeout)


class NodeWorker(RepeatTimer, Generic[IN, OUT]):
    def __init__(
        self,
        node: 'BaseNode[IN, OUT]',
        buff_size: int = 3,
        timeout: float | None = 1,
        retry: int = 0,
        standalone: bool = False,
        name: str = 'NODE',
        interval: int = 0,
        *args,
        **kwargs,
    ):
        super().__init__(interval, *args, **kwargs)
        self.buff_size = buff_size
        self.name = name
        self.upstream: Queue[IN] | None = None
        self.downstream: Queue[OUT] | None = None
        self.__node = node
        self.__logger = log.getLogger(self.__node.__class__.__name__)
        self.__upstream_nodes = list['NodeWorker[Any, IN]']()
        self.__downstream_nodes = list['NodeWorker[OUT, Any]']()
        self.__timeout = timeout
        self.__retry = retry
        self.__standalone = standalone
        self.__underflow: int = 0
        self.__overflow: int = 0

        node.bind_event(self.event)

    def __str__(self) -> str:
        return f'{self.__node.__class__.__name__}({self.name})'

    def __or__(self, other: 'NodeWorker[OUT, Any]') -> 'NodeWorker[OUT, Any]':
        if self.downstream is None and other.upstream is None:
            # None | None
            stream = Queue[OUT](max(self.buff_size, other.buff_size))
            self.downstream = stream
            other.upstream = stream
        elif self.downstream is None and other.upstream is not None:
            # None | buffer
            self.downstream = other.upstream
        elif self.downstream is not None and other.upstream is None:
            # buffer | None
            other.upstream = self.downstream
        elif self.downstream is other.upstream:
            # buffer == buffer
            pass
        else:
            raise RuntimeError('Stream connect conflict.')

        self.append_downstream(other)
        other.append_upstream(self)

        return other

    def __mul__(self, other):
        self.__or__(other)
        return self

    def __enter__(self):
        super().__enter__()
        self.__node.__enter__()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            return self.__node.__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.release()
            super().__exit__(exc_type, exc_val, exc_tb)

    @property
    def standalone(self) -> bool:
        """Whether this node ignores upstream/downstream shutdown.

        When ``True``, the node will not stop itself when upstream or downstream nodes
        become inactive or when queue operations time out.

        Returns:
            bool: ``True`` if the node is standalone, ``False`` otherwise.
        """
        return self.__standalone

    @property
    def logger(self) -> log.Logger:
        return self.__logger

    def process(self, item: IN | None) -> OUT | None:
        return self.__node.process(item)

    def routine(self):
        input_item = self.get()
        self.__node.execute_before_routine_hooks()
        output_item = self.process(input_item)
        self.__node.execute_after_routine_hooks()
        self.put(output_item)

    def start(self) -> None:
        if self.upstream is None and self.downstream is None:
            raise RuntimeError(f'{self} upstream/downstream queue are None')

        return super().start()

    def get(self) -> IN | None:
        """Get one item from upstream.

        Returns:
            IN | None: Item from upstream, or ``None`` if unavailable.
        """
        if self.upstream is None:
            return

        try:
            item = self.upstream.get(block=True, timeout=self.__timeout)
            self.__underflow = 0

            return item
        except Empty:
            if self.standalone:
                return

            if not self.is_upstream_active():
                self.__logger.warning(f'{self} End of upstream')
                self.close()
                return

            self.__underflow += 1

            if not self.__underflow > self.__retry:
                return

            self.__logger.warning(
                f'{self} upstream underflow '
                f'{self.__timeout} sec * {self.__underflow} times'
            )
            self.close()

    def put(self, item: OUT | None):
        """Put one item into downstream.

        Args:
            item (OUT | None): Item to enqueue. ``None`` is ignored.
        """
        if item is None or self.downstream is None:
            return

        try:
            self.downstream.put(item, block=True, timeout=self.__timeout)
            self.__overflow = 0
        except Full:
            if self.standalone:
                return

            if not self.is_downstream_active():
                self.__logger.warning(f'{self} end of downstream')
                self.close()
                return

            self.__overflow += 1

            if not self.__overflow > self.__retry:
                return

            self.__logger.warning(
                f'{self} output buffer overflow '
                f'{self.__timeout} sec * {self.__overflow} times'
            )
            self.close()

    def is_upstream_active(self) -> bool:
        """Check whether any upstream node is still active.

        Returns:
            bool: ``True`` if upstream work may continue, else ``False``.
        """
        with self.lock:
            upstreams = tuple(self.__upstream_nodes)

        for upstream in upstreams:
            if upstream.is_active():
                return True

        return False

    def is_downstream_active(self) -> bool:
        """Check whether any downstream node is still active.

        Returns:
            bool: ``True`` if downstream can still consume items, else ``False``.
        """
        with self.lock:
            downstreams = tuple(self.__downstream_nodes)

        for downstream in downstreams:
            if downstream.is_active():
                return True

        return False

    def release(self):
        """Disconnect linked nodes and clear local queue references."""
        with self.lock:
            upstreams = tuple(self.__upstream_nodes)
            downstreams = tuple(self.__downstream_nodes)
            self.__upstream_nodes.clear()
            self.__downstream_nodes.clear()

        for upstream in upstreams:
            upstream.remove_downstream(self)

        for downstream in downstreams:
            downstream.remove_upstream(self)

        self.upstream = None
        self.downstream = None

    def append_upstream(self, node: 'NodeWorker[Any, IN]') -> bool:
        """Register an upstream node.

        Args:
            node (Node[Any, IN]): Node to add as an upstream dependency.

        Returns:
            bool: ``True`` if added, ``False`` if already present or inactive.
        """
        with self.lock:
            if node in self.__upstream_nodes or not self.is_active():
                return False

            self.__upstream_nodes.append(node)
            return True

    def append_downstream(self, node: 'NodeWorker[OUT, Any]') -> bool:
        """Register a downstream node.

        Args:
            node (Node[OUT, Any]): Node to add as a downstream dependency.

        Returns:
            bool: ``True`` if added, ``False`` if already present or inactive.
        """
        with self.lock:
            if node in self.__downstream_nodes or not self.is_active():
                return False

            self.__downstream_nodes.append(node)
            return True

    def remove_upstream(self, node: 'NodeWorker[Any, IN]') -> bool:
        """Remove an upstream node.

        Args:
            node (Node[Any, IN]): Upstream node to remove.

        Returns:
            bool: ``True`` if removed, ``False`` if not found.
        """
        with self.lock:
            try:
                self.__upstream_nodes.remove(node)
                return True
            except ValueError:
                return False

    def remove_downstream(self, node: 'NodeWorker[OUT, Any]') -> bool:
        """Remove a downstream node.

        Args:
            node (Node[OUT, Any]): Downstream node to remove.

        Returns:
            bool: ``True`` if removed, ``False`` if not found.
        """
        with self.lock:
            try:
                self.__downstream_nodes.remove(node)
                return True
            except ValueError:
                return False

    def build_upstream_queue(self):
        """Create the upstream queue if it is not already set."""
        if self.upstream is None:
            self.upstream = Queue[IN](self.buff_size)

    def build_downstream_queue(self):
        """Create the downstream queue if it is not already set."""
        if self.downstream is None:
            self.downstream = Queue[OUT](self.buff_size)


class Node(BaseNode[IN, OUT]):
    def to_worker(
        self,
        buff_size: int = 3,
        timeout: float | None = 1,
        retry: int = 0,
        standalone: bool = False,
        name: str = 'NODE',
        interval: int = 0,
    ) -> NodeWorker[IN, OUT]:
        """Create a worker that executes this node.

        Args:
            buff_size (int, optional): Maximum number of queued items. Defaults to 3.
            timeout (float | None, optional): Queue operation timeout in seconds.
                Defaults to 1.
            retry (int, optional): Failed queue operations allowed before stopping.
                Defaults to 0.
            standalone (bool, optional): Whether to ignore linked node shutdown.
                Defaults to False.
            name (str, optional): Worker name. Defaults to 'NODE'.
            interval (int, optional): Delay between processing iterations in seconds.
                Defaults to 0.

        Returns:
            NodeWorker[IN, OUT]: Worker configured to execute this node.
        """
        return NodeWorker[IN, OUT](
            node=self,
            buff_size=buff_size,
            timeout=timeout,
            retry=retry,
            standalone=standalone,
            name=name,
            interval=interval,
        )
