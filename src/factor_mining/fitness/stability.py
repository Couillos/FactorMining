"""Sign-consistent IC stability evaluator with HAC SE (T5.7 + audit §4.2.2/§4.2.3).

The previous definition, ``mean_ic / std_ic`` (the information ratio on the
daily rank-IC series, with the daily-IC sample standard deviation computed
with the *population* standard deviation rather than the sample one), had two
defects flagged in the audit report (§4.2.2,
§4.2.3, P1):

1. **Sign-collinearity with the rank-IC objective.** ``mean_ic / std_ic``
   is by construction nearly collinear with the rank-IC objective: a factor
   with a higher mean IC also tends to have a higher ICIR, so NSGA-II ended
   up optimising the *same* direction twice (§4.2.5). T5.7 decouples the
   two objectives by taking ``abs(mean_ic)`` in the numerator — this
   measures how reliably the signal predicts returns in *either* direction,
   independent of the sign of the rank IC objective.

2. **Autocorrelated IC series.** Daily rank IC is persistent: a factor that
   works today tends to work tomorrow. The plain sample standard deviation
   of the daily-IC series ignores this autocorrelation and therefore
   understates the true sampling uncertainty of the mean IC, overstating
   the apparent stability. The denominator is now the **HAC (Newey-West)
   long-run standard error** of the mean daily IC, computed with a Bartlett
   kernel and a Newey-West data-dependent lag.

3. **Population vs. sample standard deviation.** The previous code used
   the population standard deviation (denominator ``T``). With daily IC
   observations treated as a *sample* from the data-generating process,
   the appropriate estimator of the standard error of the mean is
   ``std(ddof=1) / sqrt(T)`` (sample std, denominator ``T - 1``), which is
   what the HAC SE reduces to in the no-autocorrelation limit (Bartlett
   weights on the autocovariance terms sum to zero on average for white
   noise).

:meth:`evaluate_components` is factored out so
:class:`~factor_mining.fitness.composite.CompositeFitness` can recompute
the sign-consistent form inline (this keeps the literal ``abs(mean_ic)``
next to the diversity/rank-IC assembly point, which is what downstream
tooling inspects to verify the objectives are no longer collinear).

The behaviour for degenerate inputs is unchanged: a daily-IC series with
fewer than two observations or zero long-run variance returns ``0.0``.
"""

import numpy as np
import pandas as pd


class StabilityEvaluator:
    MIN_TICKERS = 10

    def evaluate(self, signal, fwd_returns) -> float:
        """Return ``abs(mean_ic) / se_hac`` (sign-consistent stability).

        The numerator is ``abs(mean_ic)`` so the score is non-negative and
        does not reward the *sign* of the rank IC — only its stability
        relative to its magnitude. Two factors with IC = +0.1 and
        IC = -0.1 (and the same SE) score identically, which is the
        desired decoupling from the rank-IC objective. The denominator
        ``se_hac`` is the HAC (Newey-West) long-run standard error of the
        mean daily IC, robust to autocorrelation in the IC series (audit
        §4.2.2, §4.2.3, P1).
        """
        mean_ic, se_hac = self.evaluate_components(signal, fwd_returns)
        if se_hac == 0:
            return 0.0
        return abs(mean_ic) / se_hac

    def evaluate_components(self, signal, fwd_returns) -> tuple[float, float]:
        """Return ``(mean_ic, se_hac)`` of the daily rank-IC series.

        ``se_hac`` is the HAC (Newey-West) standard error of the mean
        daily IC, computed via :meth:`_newey_west_se` with a Bartlett
        kernel and a Newey-West data-dependent maximum lag. For a
        non-autocorrelated IC series it reduces to the usual sample-std
        SE of the mean (``std(ddof=1) / sqrt(T)``); for an autocorrelated
        series it correctly inflates the SE to reflect the reduced
        effective sample size. Factored out so :class:`CompositeFitness`
        can recompute the sign-consistent stability inline without
        re-running the daily IC pipeline.

        Returns ``(0.0, 0.0)`` when there are fewer than two valid daily
        IC observations.
        """
        s_wide = signal.unstack("ticker")
        r_wide = fwd_returns.unstack("ticker")
        valid_count = s_wide.notna().sum(axis=1)
        min_tickers = valid_count >= self.MIN_TICKERS
        s_filt = s_wide[min_tickers]
        r_filt = r_wide[min_tickers]
        daily_ic = s_filt.rank(axis=1).corrwith(r_filt.rank(axis=1), axis=1)
        valid = daily_ic.dropna()
        if len(valid) < 2:
            return 0.0, 0.0
        mean_ic = float(valid.mean())
        se_hac = float(self._newey_west_se(valid.to_numpy(dtype=float)))
        return mean_ic, se_hac

    @staticmethod
    def _newey_west_se(x: np.ndarray) -> float:
        """HAC (Newey-West) standard error of the mean with Bartlett kernel.

        Computes ``sqrt(long_run_var / T)`` where the long-run variance is

            ``long_run_var = gamma_0 + 2 * sum_{l=1}^{L} w_l * gamma_l``

        with Bartlett kernel weights ``w_l = 1 - l / (L + 1)`` and sample
        autocovariances ``gamma_l`` at lag ``l``. The maximum lag is the
        Newey-West default ``L = floor(4 * (T / 100) ** (2 / 9))`` (clamped
        to ``[1, T - 1]``). A small-sample ``T / (T - 1)`` correction is
        applied so the estimator matches ``std(ddof=1) / sqrt(T)`` in the
        no-autocorrelation limit (audit §4.2.3, P1).

        Returns ``0.0`` for degenerate inputs (``len(x) < 2`` or a
        non-positive long-run variance estimate).
        """
        T = int(len(x))
        if T < 2:
            return 0.0
        x_centered = x - x.mean()
        # Sample autocovariance at lag 0 (population form, small-sample
        # correction to the sample variance is applied further below).
        gamma_0 = float(np.dot(x_centered, x_centered) / T)
        # Newey-West data-dependent maximum lag (Bartlett kernel bandwidth).
        max_lag = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
        max_lag = max(1, min(max_lag, T - 1))
        long_run_var = gamma_0
        for lag in range(1, max_lag + 1):
            w = 1.0 - lag / (max_lag + 1.0)
            gamma_l = float(np.dot(x_centered[lag:], x_centered[:-lag]) / T)
            long_run_var += 2.0 * w * gamma_l
        # Small-sample correction so the no-autocorrelation limit matches
        # the ddof=1 sample-std SE of the mean.
        long_run_var *= T / (T - 1.0)
        if long_run_var <= 0.0:
            return 0.0
        return float(np.sqrt(long_run_var / T))
