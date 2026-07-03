import numpy as np
import pandas as pd


class LongShortPortfolio:
    """Long/short equal-weight portfolio constructed from a cross-sectional signal.

    The portfolio is rebalanced at each date in the panel: the top ``decile``
    fraction of tickers (by signal value) receives equal long weights, and the
    bottom ``decile`` fraction receives equal short weights. The book is
    dollar-neutral by construction when the long and short buckets contain the
    same number of names.
    """

    def __init__(self, decile: float = 0.20):
        self.decile = decile

    def construct(self, signal) -> pd.Series:
        """Construct long/short weights for every (date, ticker) row in ``signal``.

        Parameters
        ----------
        signal : pd.Series
            MultiIndexed by ``(date_utc, ticker)``.

        Returns
        -------
        pd.Series
            Equal-weight long/short weights aligned to ``signal.index``
            (MultiIndex ``date_utc, ticker``). Sum of weights per date is 0
            (market-neutral) whenever the long and short buckets have equal
            cardinality. Returning a Series (rather than a raw numpy array)
            keeps the date/ticker index attached so downstream code can compute
            per-date turnover and apply transaction-cost drag.
        """
        dates = signal.index.get_level_values("date_utc").unique()
        weights_arr = np.zeros(len(signal), dtype=float)
        for d in dates:
            mask = signal.index.get_level_values("date_utc") == d
            idx = np.where(mask)[0]
            s = signal.iloc[idx]
            thresh = s.quantile(1 - self.decile)
            long_mask = s >= thresh
            thresh_low = s.quantile(self.decile)
            short_mask = s <= thresh_low
            n_long = long_mask.sum()
            n_short = short_mask.sum()
            if n_long > 0:
                weights_arr[idx[long_mask.values]] = 1.0 / n_long
            if n_short > 0:
                weights_arr[idx[short_mask.values]] = -1.0 / n_short
        return pd.Series(weights_arr, index=signal.index, name="weight", dtype=float)

    def rebalance(self, signal, date) -> np.ndarray:
        mask = signal.index.get_level_values("date_utc") == date
        s = signal.loc[mask]
        thresh = s.quantile(1 - self.decile)
        thresh_low = s.quantile(self.decile)
        weights = np.zeros(len(s))
        long = s >= thresh
        short = s <= thresh_low
        if long.sum() > 0:
            weights[long] = 1.0 / long.sum()
        if short.sum() > 0:
            weights[short] = -1.0 / short.sum()
        return weights

    def decile_returns(self, signal, fwd_returns, n_deciles: int = 10) -> pd.DataFrame:
        """Compute returns per decile (D1=bottom 10%, D10=top 10%).

        For each cross-section (``date_utc``) the signal is bucketed into
        ``n_deciles`` equal-population quantiles and the mean forward return
        within each bucket is reported. This is the standard Fama-MacBeth
        monotonicity view complementing the long/short spread returned by
        :meth:`construct`.

        Parameters
        ----------
        signal : pd.Series
            MultiIndexed by ``(date_utc, ticker)``.
        fwd_returns : pd.Series
            MultiIndexed by ``(date_utc, ticker)`` and aligned to ``signal``.
        n_deciles : int, default 10
            Number of equal-population quantile buckets.

        Returns
        -------
        pd.DataFrame
            ``n_deciles`` columns named ``D1..D{n_deciles}`` (D1 = lowest
            signal value, D{n_deciles} = highest) indexed by ``date_utc``.
            Rows where the cross-section has fewer than ``n_deciles`` tickers
            contain ``NaN``.
        """
        dates = signal.index.get_level_values("date_utc").unique()
        decile_returns = {f"D{i + 1}": [] for i in range(n_deciles)}
        date_list = []
        for d in dates:
            s = signal.xs(d, level="date_utc").dropna()
            f = fwd_returns.xs(d, level="date_utc").reindex(s.index)
            if len(s) < n_deciles:
                for i in range(n_deciles):
                    decile_returns[f"D{i + 1}"].append(np.nan)
                date_list.append(d)
                continue
            # Assign deciles (0=lowest, n_deciles-1=highest). ``duplicates="drop"``
            # collapses tied edges; with continuous signals this is a no-op but it
            # keeps the method robust to discrete/constant signals.
            deciles = pd.qcut(s, n_deciles, labels=False, duplicates="drop")
            for i in range(n_deciles):
                mask = deciles == i
                if mask.any():
                    decile_returns[f"D{i + 1}"].append(f[mask].mean())
                else:
                    decile_returns[f"D{i + 1}"].append(np.nan)
            date_list.append(d)
        return pd.DataFrame(decile_returns, index=pd.Index(date_list, name="date_utc"))
