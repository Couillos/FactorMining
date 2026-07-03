"""Bloat-control utilities for the GP layer.

Tree-size capping is implemented inline in :class:`NSGA2Engine` via DEAP's
``gp.staticLimit`` decorator on ``toolbox.mate`` and ``toolbox.mutate``
(see ``engine/nsga2.py`` — T5.3). The previous size-limit helper in this
module was dead code that returned a plain predicate rather than a real
DEAP decorator, so it has been removed in favour of the direct
``gp.staticLimit`` call.

This module keeps the optional double-tournament selector, which is an
alternative parsimony-pressure technique left available for future use.
"""

from deap import gp


def double_tournament_selector(toolbox, individuals, k, parsimony: float = 1.4):
    """Select ``k`` individuals using DEAP's double-tournament scheme.

    Combines a fitness tournament with a parsimony (size) tournament so
    that, all else equal, smaller trees are preferred. ``parsimony`` of
    ``0`` disables the size tournament.
    """
    return gp.selDoubleTournament(
        individuals=individuals,
        k=k,
        fitness_size=2,
        parsimony_size=2 if parsimony > 0 else 0,
        parsimony_factor=parsimony,
    )
