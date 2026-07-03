"""Combinatorial Purged Cross-Validation (CPCV).

Implements López de Prado's CPCV with purging and embargo to prevent
label leakage from forward-return windows that span the train/test
boundary between contiguous groups.

References
----------
López de Prado, M. (2018). *Advances in Financial Machine Learning*,
Chapter 7 — Cross-Validation in Finance.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Iterator


class CombinatorialPurgedCV:
    """Combinatorial Purged Cross-Validation splitter.

    Divides ``n_rows`` observations into ``n_groups`` contiguous groups,
    then yields every combination of ``k`` test groups (C(n_groups, k)
    splits). For each split, training rows are removed when their
    forward-return window would overlap the test set:

    * **Purge** — the last ``purge_window`` rows of a training group that
      immediately precedes a test group (their forward return reaches
      into the test rows).
    * **Embargo** — the first ``embargo`` rows of a training group that
      immediately follows a test group (post-test label leakage).

    Parameters
    ----------
    n_groups : int, default 10
        Number of contiguous groups into which the dataset is divided.
    k : int, default 2
        Number of test groups per split (must satisfy
        ``1 <= k < n_groups``).
    purge_window : int, default 7
        Rows removed from the trailing edge of a training group that
        precedes a test group. The default of 7 matches the standard
        forward-return horizon and prevents label leakage.
    embargo : int, default 0
        Rows removed from the leading edge of a training group that
        follows a test group.
    """

    def __init__(
        self,
        n_groups: int = 10,
        k: int = 2,
        purge_window: int = 7,
        embargo: int = 0,
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be >= 2")
        if not (1 <= k < n_groups):
            raise ValueError("k must satisfy 1 <= k < n_groups")
        if purge_window < 0:
            raise ValueError("purge_window must be >= 0")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.n_groups = int(n_groups)
        self.k = int(k)
        # Default purge_window >= 7 to cover the standard 7-day forward
        # return horizon and prevent label leakage across splits.
        self.purge_window = int(purge_window)
        self.embargo = int(embargo)

    def split(
        self, n_rows: int | None = None
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(train_indices, test_indices)`` for each C(n_groups, k) split.

        Parameters
        ----------
        n_rows : int, optional
            Total number of rows in the dataset. Returned indices are
            in ``[0, n_rows)``. Defaults to ``n_groups`` (one row per
            group) for backward compatibility with group-index callers.

        Yields
        ------
        (train_indices, test_indices) : tuple[list[int], list[int]]
            Row indices for the training and test sets. The two sets are
            guaranteed to be disjoint, and train rows whose forward-return
            window would leak labels across the train/test boundary are
            purged/embargoed.
        """
        if n_rows is None:
            # Backward-compat: one row per group so row indices match
            # group indices for legacy callers that did cv.split().
            n_rows = self.n_groups
        if n_rows < self.n_groups:
            raise ValueError(
                f"n_rows ({n_rows}) must be >= n_groups ({self.n_groups})"
            )

        groups = range(self.n_groups)
        # Even-ish group boundaries so every row belongs to exactly one
        # group, including any remainder when n_rows % n_groups != 0.
        bounds = [i * n_rows // self.n_groups for i in range(self.n_groups + 1)]

        for test_groups in itertools.combinations(groups, self.k):
            test_set = set(test_groups)
            test_indices: list[int] = []
            for g in test_groups:
                test_indices.extend(range(bounds[g], bounds[g + 1]))

            train_indices: list[int] = []
            for g in groups:
                if g in test_set:
                    continue
                start = bounds[g]
                end = bounds[g + 1]
                # Purge: drop trailing rows of a training group whose
                # forward-return window would overlap the immediately-
                # following test group.
                if (g + 1) in test_set:
                    end -= self.purge_window
                # Embargo: drop leading rows of a training group that
                # immediately follows a test group, to prevent post-test
                # label leakage.
                if (g - 1) in test_set:
                    start += self.embargo
                lo = max(0, start)
                hi = min(n_rows, end)
                if lo < hi:
                    train_indices.extend(range(lo, hi))
            yield train_indices, test_indices

    def n_combinations(self) -> int:
        """Return the number of splits C(n_groups, k)."""
        return comb(self.n_groups, self.k)


# Short alias used by validation scripts and external callers.
CPCV = CombinatorialPurgedCV
