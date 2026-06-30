from deap import gp
from deap.gp import PrimitiveTree
from factor_mining.gp.typed_pset import gen_safe
import random


def subtree_crossover(ind1, ind2, pset):
    return gp.cxOnePoint(ind1, ind2)


def subtree_mutation(ind, pset):
    def expr(pset=pset, type_=None):
        return gen_safe(pset, min_depth=1, max_depth=3, type_=type_)
    return gp.mutUniform(ind, expr=expr, pset=pset)


def point_mutation(ind, pset):
    new_ind = gp.mutNodeReplacement(ind, pset)
    if isinstance(new_ind, tuple):
        return new_ind
    return (new_ind,)
