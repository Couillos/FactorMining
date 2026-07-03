"""Domain-specific exception types for FactorMining."""


class LookaheadBiasError(Exception):
    """Raised when a signal/factor is detected using future-only data.

    Surfaced by the runtime lookahead guard
    (``factor_mining.fitness.lookahead_guard``) and caught in the NSGA2
    evaluate path so offending individuals are penalised with the -99
    sentinel fitness instead of crashing the run.
    """


class InvalidFormulaError(Exception):
    """Raised when a GP individual fails to compile or evaluates to an
    invalid formula (e.g. arity mismatch, unknown primitive)."""
    pass


class EmptyPanelError(Exception):
    """Raised when a data panel has no usable rows after cleaning."""
    pass


class BloatLimitExceeded(Exception):
    """Raised when a GP individual exceeds the configured depth/node budget."""
    pass
