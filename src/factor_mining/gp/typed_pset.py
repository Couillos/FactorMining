import random
from deap import gp
from deap.gp import PrimitiveSetTyped
from factor_mining.core.types import Panel, Window


def build_pset(factor_registry: dict) -> gp.PrimitiveSetTyped:
    pset = gp.PrimitiveSetTyped("main", [], Panel)
    pset.addPrimitive(lambda x, y: x + y, [Panel, Window], Panel, name="add_const")
    pset.addPrimitive(lambda x, y: x - y, [Panel, Window], Panel, name="sub_const")
    pset.addPrimitive(lambda x, y: x * y, [Panel, Panel], Panel, name="mul")
    pset.addPrimitive(lambda x, y: x / y, [Panel, Panel], Panel, name="div")
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
