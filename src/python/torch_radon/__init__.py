# select what to "export"

from .volumes import Volume2D, Volume3D
from .radon import FanBeam, ParallelBeam, ConeBeam
from .log_levels import *

try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    # Use backport for python<3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("torch_radon")
except PackageNotFoundError:
    # package is not installed
    pass
