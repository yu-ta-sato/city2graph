"""city2graph: A package for constructing graphs from geospatial dataset."""

import contextlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

from .graph import *  # noqa: F403
from .morphology import *  # noqa: F403
from .proximity import *  # noqa: F403
from .transportation import *  # noqa: F403
from .utils import *  # noqa: F403

__author__ = "Yuta Sato"

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("city2graph")
