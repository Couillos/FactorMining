from deap import gp
from factor_mining.gp.typed_pset import build_pset
from factor_mining.gp.primitives import register_primitives
from factor_mining.gp.subtree_cache import SubtreeCache


def test_pset_build():
    registry = {"MOM_1D": None, "MOM_7D": None}
    pset = build_pset(registry)
    assert isinstance(pset, gp.PrimitiveSetTyped)


def test_pset_registration():
    registry = {"MOM_1D": None, "MOM_7D": None}
    pset = build_pset(registry)
    pset = register_primitives(pset, registry)
    total = sum(len(v) for v in pset.primitives.values())
    assert total >= 11


def test_subtree_cache():
    cache = SubtreeCache()
    import types
    tree = types.SimpleNamespace()
    tree.__str__ = lambda self: "rank(MOM_1D)"
    cache.put(tree, (1.0, 2.0, 3.0))
    result = cache.get(tree)
    assert result == (1.0, 2.0, 3.0)


def test_subtree_cache_miss():
    cache = SubtreeCache()
    import types
    tree = types.SimpleNamespace()
    tree.__str__ = lambda self: "unknown"
    result = cache.get(tree)
    assert result is None
