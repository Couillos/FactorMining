"""Runtime lookahead-bias guards.

These helpers are invoked from the NSGA2 evaluate path to catch the most
common ways a GP-evolved factor can leak future information into the signal
that produces the in-sample fitness:

* ``check_signal_fwd_separation`` verifies the signal's decision dates are
  consistent with the forward-return panel (the signal must not contain
  decision dates for which the future is unknown).
* ``check_funding_lag`` verifies funding-rate-derived signals were shifted
  by at least one period (funding is only known *after* it has been set).
* ``check_rolling_winsorize`` verifies a winsorized panel was produced
  using only past/current data (per-date clipping bounds cannot be tighter
  than the per-date min/max of the original panel).

Each check raises :class:`LookaheadBiasError` on a violation. ``run_all_checks``
chains the checks together and is the single entry point used by the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from factor_mining.core.exceptions import LookaheadBiasError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decision_dates(index, decision_date_col: str = "date_utc") -> pd.Index:
    """Extract the sorted unique decision-date values from a MultiIndex."""
    if not isinstance(index, pd.MultiIndex):
        raise LookaheadBiasError(
            "Signal/fwd index must be a MultiIndex with a "
            f"'{decision_date_col}' level"
        )
    if decision_date_col not in index.names:
        raise LookaheadBiasError(
            f"Index missing required level '{decision_date_col}'; "
            f"got {list(index.names)}"
        )
    return pd.Index(index.get_level_values(decision_date_col)).unique()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_signal_fwd_separation(
    signal_index,
    fwd_index,
    decision_date_col: str = "date_utc",
) -> bool:
    """Verify signal and fwd_returns share a consistent decision-date index.

    The signal at decision date ``t`` may only depend on data observed at or
    before ``t``. The forward-return panel is indexed by the same decision
    dates but its values look into ``[t, t+h]``. Therefore the signal's
    decision dates must be a subset of the forward-return decision dates —
    if the signal contains a decision date that has no forward return (e.g.
    the last available bar), it means we are emitting a signal for a date
    whose future is unknown, which is a classic lookahead setup.
    """
    signal_dates = _decision_dates(signal_index, decision_date_col)
    fwd_dates = _decision_dates(fwd_index, decision_date_col)

    if len(signal_dates) == 0 or len(fwd_dates) == 0:
        raise LookaheadBiasError(
            "Cannot verify separation on an empty index "
            f"(signal_dates={len(signal_dates)}, fwd_dates={len(fwd_dates)})"
        )

    # Signal decision dates must not extend beyond the fwd-return panel.
    if signal_dates.max() > fwd_dates.max():
        raise LookaheadBiasError(
            f"Signal decision dates extend beyond forward-return panel "
            f"(signal_max={signal_dates.max()}, fwd_max={fwd_dates.max()})"
        )

    # Every signal decision date must have a matching forward return.
    extra = signal_dates.difference(fwd_dates)
    if len(extra) > 0:
        raise LookaheadBiasError(
            f"Signal has {len(extra)} decision date(s) with no matching "
            f"forward return (e.g. {extra[0]}); the signal is being emitted "
            f"for dates whose future is unknown"
        )
    return True


def check_funding_lag(
    signal,
    funding_column: str = "funding_rate",
) -> bool:
    """Verify funding-rate-derived signals respect a >=1-period lag.

    The factor layer (``factors/funding.py``) applies ``.shift(1)`` to the
    raw funding rate so a signal value at ``t`` only uses funding known at
    ``t-1``. This guard performs a runtime sanity check:

    * The signal must carry a ``(date_utc, ticker)`` MultiIndex — otherwise
      we cannot reason about per-ticker causality.
    * If the signal's ``name`` looks funding-derived (contains
      ``funding_column``), the first observation of every ticker must be
      ``NaN``: a ``.shift(1)`` always blanks the leading value, so a
      non-NaN leading value is a strong indicator the shift was skipped.
    """
    if signal is None:
        return True
    if not isinstance(signal, pd.Series):
        raise LookaheadBiasError(
            f"Funding-lag check requires a pd.Series signal, got {type(signal)!r}"
        )
    if not isinstance(signal.index, pd.MultiIndex):
        raise LookaheadBiasError(
            "Funding-derived signal must have a (date_utc, ticker) MultiIndex"
        )
    if "date_utc" not in signal.index.names or "ticker" not in signal.index.names:
        raise LookaheadBiasError(
            f"Funding-derived signal index must contain 'date_utc' and "
            f"'ticker' levels, got {list(signal.index.names)}"
        )

    name = signal.name
    if name and funding_column.lower() in str(name).lower():
        # Per-ticker first row must be NaN — that's the signature of a
        # .shift(1) on the raw funding stream.
        first_pos = signal.groupby(level="ticker", group_keys=False).apply(
            lambda s: s.iloc[:1]
        )
        if first_pos.notna().any():
            raise LookaheadBiasError(
                f"Signal '{name}' looks like an unshifted funding series: "
                f"non-NaN values at the first observation of at least one "
                f"ticker (expected .shift(1) to blank them)"
            )
    return True


def check_rolling_winsorize(
    winsorized_panel,
    original_panel,
    epsilon: float = 0.01,
) -> bool:
    """Verify winsorization was performed without future quantiles.

    Causal (per-date) winsorization clips each date's values to that date's
    own quantile bounds, so the winsorized values can never escape the
    per-date ``[min, max]`` envelope of the original panel. If a future-
    looking rolling window was used instead, the clipping bounds at date
    ``t`` depend on values from ``> t`` and the winsorized values can fall
    outside ``[min_t, max_t]``.
    """
    if winsorized_panel is None:
        return True
    if isinstance(winsorized_panel, pd.Series) and winsorized_panel.isna().all():
        return True
    if original_panel is None:
        return True
    if isinstance(original_panel, pd.Series) and original_panel.isna().all():
        return True

    common = winsorized_panel.index.intersection(original_panel.index)
    if len(common) == 0:
        return True

    w = winsorized_panel.loc[common]
    o = original_panel.loc[common]

    date_idx = o.index.get_level_values("date_utc")
    o_min = o.groupby(date_idx).transform("min")
    o_max = o.groupby(date_idx).transform("max")

    valid = w.notna() & o_min.notna() & o_max.notna()
    if not valid.any():
        return True

    too_low = (w < o_min - epsilon) & valid
    too_high = (w > o_max + epsilon) & valid
    n_low = int(too_low.sum())
    n_high = int(too_high.sum())
    if n_low or n_high:
        raise LookaheadBiasError(
            f"Winsorized panel escapes per-date original envelope "
            f"(low_violations={n_low}, high_violations={n_high}); "
            f"this suggests future-looking quantile bounds were used"
        )
    return True


def run_all_checks(signal, fwd_returns, panel=None) -> bool:
    """Run every lookahead-bias guard. Raises ``LookaheadBiasError`` on fail.

    Parameters
    ----------
    signal : pd.Series
        GP-evolved factor signal with a ``(date_utc, ticker)`` MultiIndex.
    fwd_returns : pd.Series
        Forward-return panel the signal is being scored against.
    panel : pd.DataFrame, optional
        Raw input panel. Reserved for callers that want to additionally
        verify a winsorized sub-panel; the engine evaluate path leaves
        this as ``None`` so the winsorize check is a no-op in the hot loop.
    """
    check_signal_fwd_separation(signal.index, fwd_returns.index)
    check_funding_lag(signal)

    # Rolling-winsorize guard: only triggered when a caller explicitly
    # passes a panel containing both the winsorized and original series.
    # In the NSGA2 evaluate path ``panel`` is None, so this stays inert
    # but keeps the single-entry-point contract from the audit report.
    if panel is not None and hasattr(panel, "columns"):
        if "winsorized" in panel.columns and "original" in panel.columns:
            check_rolling_winsorize(panel["winsorized"], panel["original"])

    return True
