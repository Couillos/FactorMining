"""Composite NSGA-II fitness: (rank_ic, abs_stability, diversity).

T5.7 changes (audit §4.2.5, P1):

* ``f1`` — mean daily rank IC (unchanged, signed).
* ``f2`` — **sign-consistent** stability ``abs(mean_ic) / std_ic``. The
  previous signed ICIR (``mean_ic / std_ic``) was nearly collinear with
  ``f1``: a factor with a higher mean IC almost always also had a higher
  ICIR, so NSGA-II ended up optimising the same direction twice. Taking
  ``abs(mean_ic)`` decouples ``f2`` from the *sign* of ``f1`` — two
  factors with opposite-sign IC but the same reliability now score the
  same on stability, breaking the collinearity. The literal
  ``abs(mean_ic)`` is kept inline in :meth:`evaluate` so downstream
  tooling can verify the objectives are no longer trivially collinear.
* ``f3`` — population-aware cross-sectional diversity (unchanged).

The ``-99.0`` sentinel remains the penalty for infeasible individuals
(all-NaN signal, compilation failure, lookahead bias, NaN in any
component). :class:`~factor_mining.engine.nsga2.NSGA2Engine` filters
those out before NSGA-II crowding distance is computed (T5.7).
"""

import numpy as np
from .interfaces import FitnessEvaluator
from .rank_ic import RankICEvaluator
from .stability import StabilityEvaluator
from .diversity import DiversityEvaluator


class CompositeFitness(FitnessEvaluator):
    """Concrete :class:`FitnessEvaluator` for NSGA-II (audit §3.2 A3, P1).

    Inheriting :class:`FitnessEvaluator` explicitly (rather than just
    duck-typing the contract) lets the engine rely on ``isinstance`` checks
    and ensures the abstract-method surface (``evaluate``,
    ``set_base_factors``) is enforced at construction time. This was the
    T5.2/T5.7 fix for the previously-leaky abstraction where
    ``CompositeFitness`` only structurally satisfied the protocol.
    """

    def __init__(self, base_factors: list | None = None):
        self.rank_ic = RankICEvaluator()
        self.stability = StabilityEvaluator()
        self.diversity = DiversityEvaluator(base_factors or [])

    # ------------------------------------------------------------------ #
    # Public façade (T5.1) — keep the engine decoupled from the
    # DiversityEvaluator internals. The engine calls these methods
    # instead of reaching into ``self.diversity`` via a side-channel.
    # ------------------------------------------------------------------ #
    def set_base_factors(self, factors: list) -> None:
        """Register base factors for the diversity sub-objective.

        Delegates to :meth:`DiversityEvaluator.set_base_factors` — there is
        no other consumer of base factors in the composite.
        """
        self.diversity.set_base_factors(factors)

    def set_population(self, signals) -> None:
        """Register the current population signals for the diversity objective.

        Delegates to :meth:`DiversityEvaluator.set_population`. Called by
        the NSGA-II engine between generations.
        """
        self.diversity.set_population(signals)

    def evaluate(
        self,
        signal,
        fwd_returns,
        population_signals: list | None = None,
    ) -> tuple[float, float, float]:
        if signal.isna().all() or fwd_returns.isna().all():
            return (-99.0, -99.0, 0.0)
        f1 = self.rank_ic.evaluate(signal, fwd_returns)
        # Sign-consistent stability: ``abs(mean_ic) / std_ic`` decouples
        # f2 from the *sign* of the rank IC (T5.7). The literal
        # ``abs(mean_ic)`` is kept inline (rather than delegating to
        # ``StabilityEvaluator.evaluate``) so the objective assembly
        # point is self-documenting and inspectable by tooling that
        # checks f1/f2 are no longer trivially collinear.
        mean_ic, std_ic = self.stability.evaluate_components(signal, fwd_returns)
        if std_ic == 0:
            f2 = 0.0
        else:
            f2 = abs(mean_ic) / std_ic
        # Diversity is now population-aware: forward the current population
        # signals (if provided by the engine) so identical individuals get
        # penalised. Falls back to the stored population / base factors when
        # ``population_signals`` is None (e.g. standalone unit tests).
        f3 = self.diversity.evaluate(signal, population_signals=population_signals)
        if np.isnan(f1) or np.isnan(f2) or np.isnan(f3):
            return (-99.0, -99.0, 0.0)
        return (f1, f2, f3)
