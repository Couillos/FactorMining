"""Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

Reference
---------
Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
Journal of Portfolio Management, 40(5), 94-107.

The Deflated Sharpe Ratio (DSR) is the probability that the observed Sharpe
ratio is greater than the *expected maximum* Sharpe ratio one would obtain
from ``n_trials`` independent, skill-free (SR=0) trials. It corrects the
naïve Sharpe ratio for two things:

1. **Higher moments of returns** — via the Lo (2002) / BLP14 SR-estimator
   variance::

       σ²_SR = (1 - γ₃·SR + ((γ₄-1)/4)·SR²) / (n-1)

   where γ₃ is skewness, γ₄ is Pearson kurtosis (gaussian = 3), SR is the
   per-observation Sharpe ratio, and n is the number of observations.

2. **Multiple-testing inflation** — via the expected maximum of
   ``n_trials`` iid standard normals, approximated with the
   Euler-Mascheroni constant γ ≈ 0.5772::

       E[max(Z₁..Z_N)] ≈ (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))

The DSR p-value is then::

    DSR = Φ( (SR_obs - E[max SR]) / σ_SR )
        = Φ( SR_obs/σ_SR - E[max(Z)] )
"""
import numpy as np
from scipy.stats import norm

# Euler-Mascheroni constant γ ≈ 0.5772156649 — used in the
# expected-max-of-N-Normals correction (Bailey & López de Prado, 2014, Eq. 7).
EULER_MASCHERONI = 0.5772156649

# Periods per year used to de-annualize the observed Sharpe ratio before
# plugging it into the BLP14 per-observation variance formula. Matches
# `backtest.metrics.sharpe` (daily crypto returns → sqrt(365)).
ANNUALIZATION_FACTOR = 365.0


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float,
    kurtosis: float,
) -> float:
    """Deflated Sharpe Ratio p-value (Bailey & López de Prado, 2014).

    Parameters
    ----------
    observed_sr : float
        Annualized Sharpe ratio of the strategy under test.
    n_obs : int
        Number of (non-overlapping) return observations used to compute
        ``observed_sr``.
    n_trials : int
        Number of independent strategy trials — i.e. the multiple-testing
        burden. For a GP search a sensible choice is
        ``pop_size * n_gen + len(pareto)``.
    skew : float
        Sample skewness of the return series (e.g. ``scipy.stats.skew``).
    kurtosis : float
        Sample **excess (Fisher)** kurtosis of the return series
        (e.g. ``scipy.stats.kurtosis(fisher=True)``; gaussian = 0).

    Returns
    -------
    float
        DSR p-value in [0, 1]. Higher = more confident the observed SR is
        not an artefact of multiple testing after correcting for
        skew/kurtosis. Returns 0.0 on degenerate input (``n_obs < 2``,
        ``n_trials < 1``, or non-positive SR variance).
    """
    # Guard against degenerate inputs.
    if n_obs < 2 or n_trials < 1:
        return 0.0

    # De-annualize SR: the BLP14 / Lo (2002) variance formula is derived
    # for the per-observation Sharpe ratio. The caller passes an annualized
    # SR (e.g. from `backtest.metrics.sharpe` with sqrt(365)); we convert
    # it back to per-observation units by dividing by sqrt(periods/year).
    sr_per_obs = float(observed_sr) / np.sqrt(ANNUALIZATION_FACTOR)

    # Convert Fisher kurtosis (gaussian = 0) to Pearson kurtosis
    # (gaussian = 3) so the BLP14 formula `(γ₄ - 1)/4 · SR²` reduces to
    # Lo's `(1 + 0.5·SR²)/(n-1)` for normal returns.
    kurt_pearson = float(kurtosis) + 3.0
    skew_f = float(skew)

    # σ²_SR — Lo (2002) / Bailey & López de Prado (2014) variance of the
    # Sharpe-ratio estimator.
    sr_variance = (
        1.0
        - skew_f * sr_per_obs
        + (kurt_pearson - 1.0) / 4.0 * sr_per_obs ** 2
    ) / (n_obs - 1)
    if sr_variance <= 0.0:
        return 0.0
    sigma_sr = np.sqrt(sr_variance)

    # Expected maximum of `n_trials` iid standard normals — closed-form
    # approximation using the Euler-Mascheroni constant γ ≈ 0.5772.
    gamma = EULER_MASCHERONI
    e_z_max = (
        (1.0 - gamma) * norm.ppf(1.0 - 1.0 / n_trials)
        + gamma * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    )
    # Expected max SR (in per-observation units) under the null hypothesis
    # SR_0 = 0.
    expected_max_sr = sigma_sr * e_z_max

    # DSR = P(SR_random_max < SR_observed | H0)
    dsr_z = (sr_per_obs - expected_max_sr) / sigma_sr
    return float(norm.cdf(dsr_z))


def deflated_sharpe(
    observed_sr: float,
    n_trials: int,
    sr_variance: float,  # noqa: ARG001 — kept for backward-compat; recomputed internally
    n_obs: int,
) -> float:
    """Backward-compatible wrapper around :func:`deflated_sharpe_ratio`.

    Assumes Gaussian returns (``skew = 0``, Fisher ``kurtosis = 0``). The
    legacy ``sr_variance`` argument is **ignored** — the SR variance is
    recomputed from the BLP14 formula. New call sites should call
    :func:`deflated_sharpe_ratio` directly with empirically-estimated
    skew and kurtosis.
    """
    return deflated_sharpe_ratio(
        observed_sr=observed_sr,
        n_obs=n_obs,
        n_trials=n_trials,
        skew=0.0,
        kurtosis=0.0,
    )
