"""Single, authoritative FitnessEvaluator abstraction.

This module is the *only* place where a ``FitnessEvaluator`` abstraction is
defined in the codebase (audit report §3.2 A3, P1). The previous duplicate
``Protocol`` in :mod:`factor_mining.core.interfaces` has been removed.

Design choices:

* ABC (not ``Protocol``) — concrete evaluators (``CompositeFitness``) inherit
  from it explicitly so ``isinstance`` checks and abstract-method enforcement
  work at construction time.
* ``evaluate`` is the core NSGA-II fitness hook: ``(rank_ic, stability,
  diversity)``.
* ``set_base_factors`` is the side-channel-free way for the engine to register
  base factors for diversity comparison. Previously this was only available on
  ``DiversityEvaluator`` directly, forcing callers to reach into
  ``CompositeFitness.diversity`` — a leaky implementation detail. Promoting it
  to the interface keeps the contract explicit.
"""

from abc import ABC, abstractmethod


class FitnessEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self,
        signal: "pd.Series",
        fwd_returns: "pd.Series",
    ) -> tuple[float, float, float]:
        ...

    @abstractmethod
    def set_base_factors(self, factors: list) -> None:
        ...
