from deap import gp
from copy import deepcopy
from factor_mining.factors.primitives import rank, zscore, winsor, ts_mean, ts_std, delta, ts_rank
from factor_mining.core.types import Panel, Window


def register_primitives(pset: gp.PrimitiveSetTyped, factor_names: list[str] | None = None) -> gp.PrimitiveSetTyped:
    pset.addPrimitive(rank, [Panel], Panel, name="rank")
    pset.addPrimitive(zscore, [Panel], Panel, name="zscore")
    pset.addPrimitive(winsor, [Panel], Panel, name="winsor")
    pset.addPrimitive(ts_mean, [Panel, Window], Panel, name="ts_mean")
    pset.addPrimitive(ts_std, [Panel, Window], Panel, name="ts_std")
    pset.addPrimitive(delta, [Panel, Window], Panel, name="delta")
    pset.addPrimitive(ts_rank, [Panel, Window], Panel, name="ts_rank")

    for name in (7, 14, 30, 90):
        pset.addTerminal(name, Window, name=f"W_{name}")

    if factor_names:
        for name in factor_names:
            pset.addTerminal(None, Panel, name=name)

    return pset
