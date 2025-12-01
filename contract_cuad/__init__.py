"""Contract CUAD classification package."""

from importlib.metadata import version, PackageNotFoundError

__all__ = [
    "__version__",
]

try:
    __version__ = version("contract-cuad")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"
