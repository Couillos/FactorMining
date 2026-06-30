from .primitives import winsor, zscore, neutralize, rank


def canonical_pipeline(panel, category_dummies=None):
    x = winsor(panel)
    x = zscore(x)
    if category_dummies is not None:
        x = neutralize(x, category_dummies)
    x = rank(x)
    return x
