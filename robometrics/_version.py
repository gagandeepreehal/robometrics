"""Package version."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("robometrics")
except PackageNotFoundError:
    __version__ = "0.2.0"
