from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Union

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

    def __or__(self, other: 'Element | str') -> 'Element':
        other = self.as_element(other)

        if len(self._sinks):
            self._sinks[0].__or__(other)
        else:
            self.__mul__(other)

        return self

    def __mul__(self, other: 'Element | str') -> 'Element':
        other = self.as_element(other)
        self._sinks.append(other)
        other._srcs.append(self)

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
    def _srcs(self) -> LE:
        if self.__srcs is None:
            self.__srcs = LE()

        return self.__srcs

    @property
    def _sinks(self) -> LE:
        if self.__sinks is None:
            self.__sinks = LE()

        return self.__sinks

    def print(self):
        print(self._build())

    def build(self):
        return self._build()

    def _build(
        self,
        skip_element: Union['Element', None] = None,
        pool: set[str] | None = None,
    ) -> str:
        pool = pool or set[str]()
        links = list[str]()
        self_element = self.__build_element()
        element_appeneded = False

        if self._is_require_name():
            self_element = f'{self_element} name={self._fetch_name(pool)}'

        for src in self._srcs:
            if src is skip_element:
                continue

            src_element = f'{src._build(self, pool)}'

            if src._has_multi_sink():
                src_element = f'{src_element} {src._fetch_name(pool)}.'

            if self._has_multi_src():
                links.append(f'{src_element} ! {self._fetch_name(pool)}.')
                continue

            links.append(f'{src_element} ! {self_element}')
            element_appeneded = True

        if not element_appeneded:
            links.append(self_element)

        for sink in self._sinks:
            if sink is skip_element:
                continue

            if not self._has_multi_sink():
                links.append(f'! {sink._build(self, pool)}')
                continue

            links.append(f'{self._fetch_name(pool)}. ! {sink._build(self, pool)}')

        return ' '.join(links)

    def _has_multi_src(self) -> bool:
        return len(self._srcs) > 1

    def _has_multi_sink(self) -> bool:
        return len(self._sinks) > 1

    def _is_require_name(self) -> bool:
        return self._has_multi_src() or self._has_multi_sink()

    def _fetch_name(self, name_pool: set[str]) -> str:
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


@dataclass(slots=True)
class GstElement(Element):
    def as_element(self, instance: Element | str) -> Element:
        if isinstance(instance, Element):
            return instance

        return RawElement(instance)

    def get_properties(self):
        return asdict(self)
