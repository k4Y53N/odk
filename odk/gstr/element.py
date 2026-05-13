from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Union, overload

__all__ = [
    'Element',
    'RawElement',
    'GstElement',
]

LE = list['Element']


class Element(ABC):
    __srcs: LE | None = None
    __sinks: LE | None = None
    __cache_name: str | None = None

    def __str__(self):
        return self.__build_element()

    @overload
    def __or__(self, other: 'Element') -> 'Element': ...

    @overload
    def __or__(self, other: str) -> 'Element': ...

    def __or__(self, other: 'Element | str') -> 'Element':
        other = self.as_element(other)

        if len(self.sinks):
            self.sinks[0].__or__(other)
        else:
            self.__mul__(other)

        return self

    @overload
    def __mul__(self, other: 'Element') -> 'Element': ...

    @overload
    def __mul__(self, other: str) -> 'Element': ...

    def __mul__(self, other: 'Element | str') -> 'Element':
        other = self.as_element(other)

        self.sinks.append(other)
        other.srcs.append(self)

        return self

    @abstractmethod
    def as_element(self, instance: 'Element | str') -> 'Element': ...

    @abstractmethod
    def get_properties(self) -> dict[str, Any]: ...

    def T(self) -> str:
        return self.__class__.__name__.lower()

    def separate(self):
        return ' '

    @property
    def name(self) -> str:
        return self.T().upper()

    @property
    def srcs(self) -> LE:
        if self.__srcs is None:
            self.__srcs = LE()

        return self.__srcs

    @property
    def sinks(self) -> LE:
        if self.__sinks is None:
            self.__sinks = LE()

        return self.__sinks

    def print(self):
        print(self.build())

    def build(
        self,
        skip_element: Union['Element', None] = None,
        pool: set[str] | None = None,
    ) -> str:
        pool = pool or set[str]()
        links = list[str]()
        self_element = self.__build_element()
        element_appeneded = False

        if self.is_require_name():
            self_element = f'{self_element} name={self.fetch_name(pool)}'

        for src in self.srcs:
            if src is skip_element:
                continue

            src_element = f'{src.build(self, pool)}'

            if src.has_multi_sink():
                src_element = f'{src_element} {src.fetch_name(pool)}.'

            if self.has_multi_src():
                links.append(f'{src_element} ! {self.fetch_name(pool)}.')
                continue

            links.append(f'{src_element} ! {self_element}')
            element_appeneded = True

        if not element_appeneded:
            links.append(self_element)

        for sink in self.sinks:
            if sink is skip_element:
                continue

            if not self.has_multi_sink():
                links.append(f'! {sink.build(self, pool)}')
                continue

            links.append(f'{self.fetch_name(pool)}. ! {sink.build(self, pool)}')

        return ' '.join(links)

    def has_multi_src(self) -> bool:
        return len(self.srcs) > 1

    def has_multi_sink(self) -> bool:
        return len(self.sinks) > 1

    def is_require_name(self) -> bool:
        return self.has_multi_src() or self.has_multi_sink()

    def fetch_name(self, name_pool: set[str]) -> str:
        if self.__cache_name is not None:
            return self.__cache_name

        unique_name = self.name
        uid = 1

        while unique_name in name_pool:
            unique_name = f'{self.name}_{uid}'
            uid += 1

        name_pool.add(unique_name)
        self.__cache_name = unique_name

        return unique_name

    def __build_element(self) -> str:
        return self.separate().join([self.T(), *self.__build_properties()])

    def __build_properties(self) -> list[str]:
        return [
            f'{self.__clean_key(k)}={self.__clean_value(v)}'
            for k, v in self.get_properties().items()
            if v is not None
        ]

    def __clean_key(self, key: str) -> str:
        if key.startswith('_'):
            key = key[1:]

        return key.replace('_', '-')

    def __clean_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return str(value).lower()

        value = str(value).strip()

        if ' ' in value:
            return f'"{value}"'

        return value


class RawElement(Element):
    def __init__(self, E: str, **kwargs):
        self.E = E
        self.properties = kwargs

    def T(self):
        return self.E

    def as_element(self, instance: Element | str) -> Element:
        if isinstance(instance, Element):
            return instance

        return RawElement(instance)

    def get_properties(self):
        return self.properties


@dataclass
class GstElement(Element):
    def as_element(self, instance: Element | str) -> Element:
        if isinstance(instance, Element):
            return instance

        return RawElement(instance)

    def get_properties(self):
        return asdict(self)
