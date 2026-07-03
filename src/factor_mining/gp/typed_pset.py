import random
import numpy as np
from deap import gp
from deap.gp import PrimitiveSetTyped
from factor_mining.core.types import Panel, Window


def _safe_div(x, y):
    """Safe division: returns NaN where denominator is near zero.

    The previous bare division primitive produced ``+/-inf`` (or ``nan``
    via 0/0) whenever the denominator panel hit exact zeros — a frequent
    case for sparse funding/volume factors that are zero on most dates.
    By masking near-zero denominators (|y| <= 1e-10) with NaN *before*
    the division, the result is always finite-or-NaN, never ``inf``.
    Downstream rank-IC / stability / diversity evaluators already drop
    NaNs, so this preserves signal quality without leaking infinities
    into cross-sectional statistics.

    Handles both pandas ``Series`` (the canonical ``Panel`` type) and
    scalar ``y`` — the latter shows up when a constant sub-expression
    folds through DEAP's compiler.
    """
    if hasattr(y, "where"):
        # pandas Series (Panel): mask near-zero denominators with NaN.
        return x / y.where(y.abs() > 1e-10, np.nan)
    else:
        # scalar denominator
        return x / y if abs(y) > 1e-10 else np.nan


def build_pset(factor_registry: dict) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("main", [], Panel)
    pset.addPrimitive(lambda x, y: x * y, [Panel, Panel], Panel, name="mul")
    pset.addPrimitive(_safe_div, [Panel, Panel], Panel, name="div")
    pset.addPrimitive(lambda x, y: x + y, [Panel, Panel], Panel, name="add")
    pset.addPrimitive(lambda x, y: x - y, [Panel, Panel], Panel, name="sub")
    return pset


def gen_safe(pset, min_depth, max_depth, type_=None):
    if type_ is None:
        type_ = pset.ret
    height = random.randint(min_depth, max_depth)
    expr = []
    stack = [(0, type_)]
    while stack:
        depth, t = stack.pop()
        if t == Window:
            term = random.choice(pset.terminals[t])
            expr.append(term)
        elif depth >= height:
            term = random.choice(pset.terminals[t])
            expr.append(term)
        else:
            prim = random.choice(pset.primitives[t])
            expr.append(prim)
            for arg_type in reversed(prim.args):
                stack.append((depth + 1, arg_type))
    return expr
