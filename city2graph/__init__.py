"""
city2graph: A package for constructing graphs from geospatial dataset.
"""
import contextlib
from importlib.metadata import PackageNotFoundError, version

from .utils import *
from .morphological_network import *
from .graph import *

__author__ = "Yuta Sato"

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("city2graph")