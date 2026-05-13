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


@dataclass
class DecodeBin(GstElement): ...


@dataclass
class DecodeBin3(GstElement): ...


@dataclass
class UriDecodeBin(GstElement):
    uri: str


@dataclass
class UriDecodeBin3(GstElement):
    uri: str


@dataclass
class ParseBin(GstElement): ...


@dataclass
class UriSourceBin(GstElement):
    uri: str
