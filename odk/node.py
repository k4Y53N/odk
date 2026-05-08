import logging as log
from abc import ABC, abstractmethod
from queue import Empty, Full, Queue
from typing import Any, Generic, TypeVar

from .repeat_timer import RepeatTimer

__all__ = [
    'Node',
]


IN = TypeVar('IN')
OUT = TypeVar('OUT')


class Node(RepeatTimer, Generic[IN, OUT], ABC):
    def __init__(
        self,
        buff_size: int = 3,
        timeout: float | None = 1,
        retry: int | float = 0,
        standalone: bool = False,
        name: str = 'NODE',
        interval: int = 0,
        *args,
        **kwargs,
    ):
        """Initialize a processing node with queue and retry settings.

        Args:
            buff_size (int, optional): Queue capacity used when creating
                upstream/downstream buffers. Defaults to 3.
            timeout (float | None, optional): Timeout in seconds for queue
                get/put operations. ``None`` means block indefinitely.
                Defaults to 1.
            retry (int | float, optional): Maximum consecutive underflow or
                overflow attempts before stopping. Defaults to 0.
            standalone (bool, optional): If ``True``, do not stop when
                upstream/downstream stopped. Defaults to False.
            name (str, optional): Display name used in logs and ``__str__``.
                Defaults to 'NODE'.
            interval (int, optional): Delay in seconds between routine ticks.
                Passed to :class:`RepeatTimer`. Defaults to 0.
            *args: Positional arguments forwarded to :class:`RepeatTimer`.
            **kwargs: Keyword arguments forwarded to :class:`RepeatTimer`.
        """
        super().__init__(interval, *args, **kwargs)
        self.buff_size = buff_size
        self.name = name
        self.upstream: Queue[IN] | None = None
        self.downstream: Queue[OUT] | None = None
        self.__logger = log.getLogger(self.__class__.__name__)
        self.__upstream_nodes = list['Node[Any, IN]']()
        self.__downstream_nodes = list['Node[OUT, Any]']()
        self.__timeout = timeout
        self.__retry = retry
        self.__standalone = standalone
        self.__underflow: int = 0
        self.__overflow: int = 0

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.name})'

    def __or__(self, other: 'Node[OUT, Any]') -> 'Node[OUT, Any]':
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
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        super().__exit__(exc_type, exc_val, exc_tb)
        self.release()

        return False

    @abstractmethod
    def process(self, item: IN | None) -> OUT | None:
        """Transform one input item into an output item.

        Subclasses must implement this method with the node-specific
        processing logic.

        Args:
            item (IN | None): Input item from upstream, or ``None``.

        Returns:
            OUT | None: Processed output item, or ``None`` to skip enqueueing.
        """

    @property
    def standalone(self) -> bool:
        """Whether queue underflow/overflow should not stop the node.

        Returns:
            bool: ``True`` when keep-alive mode is enabled.
        """
        return self.__standalone

    @property
    def logger(self) -> log.Logger:
        return self.__logger

    def routine(self):
        input_item = self.get()
        output_item = self.process(input_item)
        self.put(output_item)

    def before(self): ...

    def after(self): ...

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

    def append_upstream(self, node: 'Node[Any, IN]') -> bool:
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

    def append_downstream(self, node: 'Node[OUT, Any]') -> bool:
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

    def remove_upstream(self, node: 'Node[Any, IN]') -> bool:
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

    def remove_downstream(self, node: 'Node[OUT, Any]') -> bool:
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
