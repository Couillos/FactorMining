import itertools
import numpy as np


class CombinatorialPurgedCV:
    def __init__(self, n_groups: int = 10, k: int = 2, purge_window: int = 0):
        self.n_groups = n_groups
        self.k = k
        self.purge_window = purge_window

    def split(self):
        groups = list(range(self.n_groups))
        for test_idx in itertools.combinations(groups, self.k):
            train_idx = [g for g in groups if g not in test_idx]
            yield train_idx, list(test_idx)

    def n_combinations(self) -> int:
        from math import comb
        return comb(self.n_groups, self.k)
