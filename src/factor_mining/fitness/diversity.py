"""Population-aware diversity objective for the NSGA-II factor mining engine.

The previous implementation compared each candidate signal only against a fixed
set of *base factors* using a single global Pearson correlation on ranks. That
had two failure modes (audit report §4.4.4, P0 #9):

1. **No population awareness** — two genetically identical GP individuals both
   scored ``diversity = 1.0`` because neither appeared in the base-factor set,
   so NSGA-II had no signal to penalise population collapse.
2. **Global correlation** — averaging over the entire ``(date, ticker)`` panel
   mixes cross-sectional information (per-date rank ordering) with time-series
   information, which is not what "diversity of alpha" means in a quant
   context. Two factors can be globally uncorrelated yet cosmetically similar
   on each trading day.

This module fixes both issues. Diversity is now

    diversity = 1 - mean(|cross-sectional Spearman rank correlation|)

where the correlation is computed **per date** (across tickers) and then
averaged over dates and over the comparison set. The comparison set is the
*current population* of surviving signals (priority order):

1. ``population_signals`` argument passed to :meth:`evaluate`
2. signals registered via :meth:`set_population`
3. base factors registered via :meth:`set_base_factors`

With this definition, two identical signals have ``|corr| = 1`` on every date,
so ``diversity ≈ 0`` — exactly the penalty we want against collapse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class DiversityEvaluator:
    """Cross-sectional, population-aware diversity evaluator.

    See module docstring for the design rationale.
    """

    #: Minimum number of non-NaN tickers required on a given date for the
    #: per-date Spearman correlation to be considered meaningful. Matches the
    #: ``MIN_TICKERS`` floor used by :mod:`factor_mining.fitness.rank_ic` so
    #: the diversity objective does not reward dates that the rank-IC
    #: objective itself would discard.
    MIN_TICKERS: int = 10

    def __init__(self, base_factors: list | None = None):
        # Store as a plain list — we accept either ``pd.Series`` (preferred,
        # has MultiIndex date_utc/ticker) or ``np.ndarray`` (will be wrapped
        # against the candidate signal's index on the fly).
        self.base_factors: list = list(base_factors) if base_factors else []
        self.population_signals: list = []

    # ------------------------------------------------------------------ #
    # Setters
    # ------------------------------------------------------------------ #
    def set_base_factors(self, factors) -> None:
        """Register base factors for diversity comparison.

        ``factors`` should be a list of ``pd.Series`` indexed by the
        ``(date_utc, ticker)`` MultiIndex so that cross-sectional correlation
        can be computed. ``np.ndarray`` inputs are tolerated and will be
        aligned against the candidate signal's index at evaluation time.
        """
        self.base_factors = list(factors) if factors else []

    def set_population(self, signals) -> None:
        """Register the current population signals.

        Called by the NSGA-II engine between generations so that each
        candidate is compared against the *surviving* population rather than
        only against the static base-factor set.
        """
        self.population_signals = list(signals) if signals else []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_series(x, ref: pd.Series | None = None) -> pd.Series:
        """Coerce a signal-like object to a ``pd.Series``.

        - ``pd.Series`` returned as-is.
        - ``pd.DataFrame`` reduced to its first column.
        - ``np.ndarray`` / list wrapped with ``ref.index`` when sizes match
          (needed for backward-compat with the old base-factor interface that
          received ``.values`` arrays).
        """
        if isinstance(x, pd.Series):
            return x
        if isinstance(x, pd.DataFrame):
            return x.iloc[:, 0]
        arr = np.asarray(x).ravel()
        if ref is not None and isinstance(ref, pd.Series) and len(arr) == len(ref):
            return pd.Series(arr, index=ref.index)
        return pd.Series(arr)

    @classmethod
    def _cross_sectional_corr(cls, s1: pd.Series, s2: pd.Series) -> float:
        """Mean absolute per-date Spearman rank correlation.

        Both signals must share a MultiIndex containing ``date_utc`` and
        ``ticker`` levels. For each date, ranks tickers and computes Pearson
        correlation on the ranks (== Spearman). Returns the mean of
        ``|daily_corr|`` across dates with at least ``MIN_TICKERS`` jointly
        valid tickers, or ``NaN`` if no date qualifies.
        """
        if not isinstance(s1, pd.Series) or not isinstance(s2, pd.Series):
            return np.nan
        names1 = s1.index.names or []
        names2 = s2.index.names or []
        if "date_utc" not in names1 or "date_utc" not in names2:
            return np.nan
        if "ticker" not in names1 or "ticker" not in names2:
            return np.nan

        # Inner-join on the MultiIndex and drop any (date, ticker) pairs
        # where either signal is NaN — per-date correlation requires paired
        # observations.
        df = pd.concat([s1.rename("s"), s2.rename("o")], axis=1).dropna()
        if df.empty:
            return np.nan

        s1_wide = df["s"].unstack("ticker")
        s2_wide = df["o"].unstack("ticker")
        # Joint per-date valid count (both must be non-NaN).
        joint_valid = s1_wide.notna() & s2_wide.notna()
        ok_dates = joint_valid.sum(axis=1) >= cls.MIN_TICKERS
        if not ok_dates.any():
            return np.nan

        # Pearson on ranks == Spearman. Rank across tickers (axis=1) for each
        # date separately.
        s1_rank = s1_wide[ok_dates].rank(axis=1)
        s2_rank = s2_wide[ok_dates].rank(axis=1)
        # corrwith(axis=1) -> per-row correlation across columns.
        daily_corr = s1_rank.corrwith(s2_rank, axis=1).dropna()
        if daily_corr.empty:
            return np.nan
        return float(abs(daily_corr.mean()))

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        signal,
        fwd_returns=None,
        population_signals: list | None = None,
    ) -> float:
        """Return ``1 - mean(|cross-sectional Spearman correlation|)``.

        Comparison set, in priority order:
          1. ``population_signals`` argument (if not ``None``)
          2. signals set via :meth:`set_population`
          3. base factors set via :meth:`set_base_factors`

        Identical signals → ``|corr| = 1`` → diversity ≈ 0.
        Uncorrelated signals → ``|corr| ≈ 0`` → diversity ≈ 1.
        Returns ``1.0`` (maximally diverse) when the comparison set is empty
        or when the signal lacks the required MultiIndex.
        """
        # 1. Pick the comparison set
        if population_signals is not None:
            others = list(population_signals)
        elif self.population_signals:
            others = list(self.population_signals)
        else:
            others = list(self.base_factors)

        if not others:
            return 1.0

        # 2. Validate the candidate signal
        if not isinstance(signal, pd.Series):
            return 1.0
        if "date_utc" not in (signal.index.names or []):
            return 1.0

        # 3. Compute |cross-sectional corr| against each "other" signal
        corrs: list[float] = []
        for other in others:
            other_series = self._as_series(other, ref=signal)
            rho = self._cross_sectional_corr(signal, other_series)
            if rho is not None and not np.isnan(rho):
                corrs.append(rho)

        if not corrs:
            return 1.0
        return float(1.0 - np.mean(corrs))
