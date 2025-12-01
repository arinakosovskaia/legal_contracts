import csv
import importlib.machinery
import re
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_pandas_stub() -> None:
    if "pandas" in sys.modules:
        return

    class _DataFrame:
        def __init__(self, rows, columns):
            self._rows = rows
            self.columns = columns

        def iterrows(self):
            for idx, row in enumerate(self._rows):
                yield idx, row

    def _read_csv(path_or_buf):
        path = Path(path_or_buf)
        with path.open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
        return _DataFrame(rows, columns)

    pandas_stub = ModuleType("pandas")
    pandas_stub.read_csv = _read_csv  # type: ignore[attr-defined]
    pandas_stub.__spec__ = importlib.machinery.ModuleSpec("pandas", loader=None)
    sys.modules["pandas"] = pandas_stub


def _install_nltk_stub() -> None:
    if "nltk" in sys.modules:
        return

    nltk_stub = ModuleType("nltk")

    class _Data:
        def find(self, *args, **kwargs):
            return True

    def _sent_tokenize(text):
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]

    def _download(*args, **kwargs):
        return True

    nltk_stub.data = _Data()
    nltk_stub.sent_tokenize = _sent_tokenize  # type: ignore[attr-defined]
    nltk_stub.download = _download  # type: ignore[attr-defined]
    nltk_stub.__spec__ = importlib.machinery.ModuleSpec("nltk", loader=None)
    sys.modules["nltk"] = nltk_stub


def _install_datasets_stub() -> None:
    if "datasets" in sys.modules:
        return

    datasets_stub = ModuleType("datasets")

    class _Dataset(list):
        @classmethod
        def from_list(cls, items):
            return cls(items)

    def _load_dataset(*args, **kwargs):
        raise RuntimeError("datasets.load_dataset stub should not be called in tests")

    datasets_stub.Dataset = _Dataset  # type: ignore[attr-defined]
    datasets_stub.load_dataset = _load_dataset  # type: ignore[attr-defined]
    datasets_stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    sys.modules["datasets"] = datasets_stub


_install_pandas_stub()
_install_nltk_stub()
_install_datasets_stub()
