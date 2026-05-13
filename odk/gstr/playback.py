from dataclasses import dataclass

from .element import GstElement

__all__ = [
    'DecodeBin',
    'DecodeBin3',
    'UriDecodeBin',
    'UriDecodeBin3',
    'ParseBin',
    'UriSourceBin',
]


@dataclass(slots=True)
class DecodeBin(GstElement): ...


@dataclass(slots=True)
class DecodeBin3(GstElement): ...


@dataclass(slots=True)
class UriDecodeBin(GstElement):
    uri: str


@dataclass(slots=True)
class UriDecodeBin3(GstElement):
    uri: str


@dataclass(slots=True)
class ParseBin(GstElement): ...


@dataclass(slots=True)
class UriSourceBin(GstElement):
    uri: str
