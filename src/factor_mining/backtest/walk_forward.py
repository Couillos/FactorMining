"""Walk-forward window splitter with embargo between IS and OOS.

For a forward-return horizon of ``H`` days, the label observed at the last IS
timestamp actually depends on prices observed at ``is_end + H``. If we set
``oos_start = is_end`` (no embargo), the last IS label leaks prices that fall
inside the OOS window — a textbook lookahead/label-leakage bug.

We therefore insert an embargo of ``embargo_days`` (default = forward return
horizon) between ``is_end`` and ``oos_start`` so that no IS label's price
window overlaps with the OOS price window.

References: audit report §4.3.6, P0 #7.
"""
from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class WalkForwardWindow:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


class WalkForwardRunner:
    """Generate walk-forward (IS, OOS) windows with an embargo gap.

    Parameters
    ----------
    is_days : int
        Length of the in-sample window in calendar days.
    oos_days : int
        Length of the out-of-sample window in calendar days.
    step_days : int
        Step between consecutive walk-forward windows in calendar days.
    fwd_horizon_days : int
        Forward-return horizon used to label the data. Used as the default
        embargo when ``embargo_days`` is not provided.
    embargo_days : int | None
        Embargo between ``is_end`` and ``oos_start``. Defaults to
        ``fwd_horizon_days`` to guarantee no label leakage from a
        ``fwd_horizon_days``-day forward return into the OOS window.

    Notes
    -----
    The gap between ``is_end`` and ``oos_start`` is always
    ``>= embargo_days >= 1``, ensuring no IS label's price window overlaps the
    OOS price window. Windows are emitted as long as a full IS window fits
    inside ``[start, end]``; the OOS tail of the last window may extend a few
    days past ``end`` so the full period is covered.
    """

    def __init__(
        self,
        is_days: int = 365,
        oos_days: int = 90,
        step_days: int = 90,
        fwd_horizon_days: int = 7,
        embargo_days: int | None = None,
    ):
        if is_days <= 0 or oos_days <= 0 or step_days <= 0:
            raise ValueError("is_days, oos_days and step_days must be positive")
        if fwd_horizon_days < 0:
            raise ValueError("fwd_horizon_days must be non-negative")
        self.is_days = is_days
        self.oos_days = oos_days
        self.step_days = step_days
        self.fwd_horizon_days = fwd_horizon_days
        # Default embargo = forward return horizon to prevent label leakage.
        # A 7-day forward return at the last IS timestamp uses prices up to
        # is_end + 7; the OOS window must therefore start at is_end + 7.
        self.embargo_days = (
            embargo_days if embargo_days is not None else fwd_horizon_days
        )
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be non-negative")

    def split(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> list[WalkForwardWindow]:
        """Generate non-overlapping walk-forward windows with embargo.

        For each window:

        * ``is_end   = is_start + is_days``
        * ``oos_start = is_end + embargo_days``  (the embargo gap)
        * ``oos_end   = oos_start + oos_days``

        The embargo guarantees ``oos_start - is_end >= embargo_days``, so a
        forward return of length ``<= embargo_days`` observed at the last IS
        timestamp cannot leak prices from the OOS window.
        """
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if end <= start:
            return []

        windows: list[WalkForwardWindow] = []
        cur = start
        # Continue while a full IS window still fits inside [start, end].
        # The OOS tail of the final window is allowed to extend slightly past
        # `end` so that the full period is covered (downstream code slices by
        # date and naturally clips to available data).
        while cur + pd.Timedelta(days=self.is_days) <= end:
            is_end = cur + pd.Timedelta(days=self.is_days)
            oos_start = is_end + pd.Timedelta(days=self.embargo_days)
            oos_end = oos_start + pd.Timedelta(days=self.oos_days)
            windows.append(
                WalkForwardWindow(
                    is_start=cur,
                    is_end=is_end,
                    oos_start=oos_start,
                    oos_end=oos_end,
                )
            )
            cur += pd.Timedelta(days=self.step_days)
        return windows

    # Backward-compatible alias (old callers used ``get_windows``).
    def get_windows(
        self,
        start: str | pd.Timestamp,
        end: str | pd.Timestamp,
    ) -> list[WalkForwardWindow]:
        """Deprecated alias for :meth:`split`."""
        return self.split(start, end)
