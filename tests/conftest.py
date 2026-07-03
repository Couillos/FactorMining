import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_panel.pkl"


def _ensure_synthetic_fixture() -> Path:
    """Create the synthetic panel pickle on disk if it is missing.

    The fixture is committed to the repo via an explicit `.gitignore` exception
    (`!tests/fixtures/synthetic_panel.pkl`), so on a healthy checkout it already
    exists. This auto-generation is a defensive fallback so that a fresh clone
    in an environment where the binary pickle did not survive the transfer
    (e.g. shallow clone, sparse checkout, LFS issues) still passes `pytest`
    without manual intervention.
    """
    if _FIXTURE_PATH.exists():
        return _FIXTURE_PATH

    _FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Make the repo root importable so `import scripts.generate_synthetic_data`
    # works regardless of where pytest is invoked from.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.generate_synthetic_data import generate_synthetic_panel

    panel = generate_synthetic_panel()
    panel.to_pickle(_FIXTURE_PATH)
    print(f"[conftest] auto-generated {_FIXTURE_PATH} ({len(panel)} rows)")
    return _FIXTURE_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_fixture():
    """Session-scoped autouse guard: materialize the synthetic panel if absent."""
    _ensure_synthetic_fixture()


@pytest.fixture(scope="session")
def synthetic_panel():
    _ensure_synthetic_fixture()
    return pd.read_pickle(_FIXTURE_PATH)


@pytest.fixture
def simple_panel():
    dates = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    tickers = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    data = np.random.default_rng(42).normal(0, 1, len(idx))
    return pd.Series(data, index=idx)


@pytest.fixture
def linear_signal():
    dates = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    tickers = [f"T{i:03d}" for i in range(20)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])
    rng = np.random.default_rng(42)
    values = rng.uniform(-1, 1, len(idx))
    return pd.Series(values, index=idx)
