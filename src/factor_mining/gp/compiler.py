from deap import gp


def compile_tree(tree, pset):
    try:
        result = gp.compile(tree, pset)
        if callable(result):
            return result
        return lambda: result
    except Exception:
        return None
