import pandas as pd
import numpy as np
from factor_mining.factors.registry import FactorRegistry


def test_registry_has_16_factors():
    registry = FactorRegistry()
    assert len(registry) == 16


def test_all_factor_names():
    registry = FactorRegistry()
    expected = {
        "MOM_1D", "MOM_7D", "MOM_30D", "MOM_90D",
        "FUNDING_RATE", "FUNDING_RATE_ZS",
        "TAKER_BUY_RATIO", "TAKER_NET_VOLUME",
        "OI_CHANGE", "OI_USD",
        "LS_RATIO", "LS_RATIO_ZS",
        "VOL_30D", "LOG_MCAP", "AMIHUD", "SKEW_30D",
    }
    assert set(registry.list()) == expected


def test_factor_compute(synthetic_panel):
    registry = FactorRegistry()
    for name in registry.list():
        factor = registry.get(name)
        result = factor.compute(synthetic_panel)
        assert isinstance(result, pd.Series)
        assert result.dtype == np.float64
        assert "date_utc" in result.index.names
        assert "ticker" in result.index.names


def test_funding_rate_lag(synthetic_panel):
    from factor_mining.factors.funding import FUNDING_RATE
    factor = FUNDING_RATE()
    result = factor.compute(synthetic_panel)
    panel_fr = synthetic_panel["funding_rate"]
    ticker = result.index.get_level_values("ticker").unique()[0]
    mask = result.index.get_level_values("ticker") == ticker
    assert result.loc[mask].iloc[0] != panel_fr.loc[mask].iloc[0]


def test_canonical_pipeline_order(synthetic_panel):
    from factor_mining.factors.transforms import canonical_pipeline
    signal = synthetic_panel["factor_00"]
    result = canonical_pipeline(signal)
    assert isinstance(result, pd.Series)
