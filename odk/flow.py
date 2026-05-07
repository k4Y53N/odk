from collections.abc import Callable
from typing import ParamSpec, Protocol, TypeVar, overload

from .node import Node

__all__ = [
    'Flow',
    'FlowHook',
]

P = ParamSpec('P')
R = TypeVar('R')


class NodeMethod(Protocol):
    @overload
    def __call__(self: Node): ...
    @overload
    def __call__(self: Node, *args, **kwargs): ...


class FlowHook:
    def __init__(self, fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class Flow:
    def __init__(self, *nodes: Node):
        prev = None

        for node in nodes:
            if prev is None:
                prev = node
                continue

            prev | node
            prev = node

        self.nodes = nodes

    def call(self, skip_keep_alive: bool, method: NodeMethod, *args, **kwargs) -> bool:
        flag = True

        for node in self.nodes:
            if skip_keep_alive and node.keep_alive:
                continue

            ret = method(node, *args, **kwargs)
            flag &= bool(ret)

        return flag

    def start(self, skip_keep_alive: bool = True):
        self.call(skip_keep_alive, Node.start)

    def close(self, skip_keep_alive: bool = True):
        self.call(skip_keep_alive, Node.close)

    def join(self, skip_keep_alive: bool = True):
        self.call(skip_keep_alive, Node.join)

    def is_active(self, skip_keep_alive: bool = True):
        return self.call(skip_keep_alive, Node.is_active)

    def add_enter_hook(self, hook: FlowHook, skip_keep_alive: bool = True):
        self.call(
            skip_keep_alive, Node.add_enter_hook, hook.fn, *hook.args, **hook.kwargs
        )

    def add_exit_hook(self, hook: FlowHook, skip_keep_alive: bool = True):
        self.call(
            skip_keep_alive,
            Node.add_exit_hook,
            hook.fn,
            *hook.args,
            **hook.kwargs,
        )

    def add_self_close_hook(self, skip_keep_alive: bool = True):
        self.add_exit_hook(FlowHook(self.close, skip_keep_alive), skip_keep_alive)
