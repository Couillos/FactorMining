from factor_mining.core.types import Panel, Window
from factor_mining.core.exceptions import LookaheadBiasError, InvalidFormulaError, EmptyPanelError, BloatLimitExceeded


def test_type_aliases():
    assert Panel is not None
    assert Window is int


def test_exceptions():
    assert issubclass(LookaheadBiasError, Exception)
    assert issubclass(InvalidFormulaError, Exception)
    assert issubclass(EmptyPanelError, Exception)
    assert issubclass(BloatLimitExceeded, Exception)


def test_tree_chromosome():
    from factor_mining.core.chromosome import TreeChromosome
    tc = TreeChromosome(tree=None, fitness=(1.0, 2.0, 3.0))
    assert tc.fitness == (1.0, 2.0, 3.0)
