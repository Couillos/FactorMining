import pandas as pd
from dataclasses import dataclass


@dataclass
class WalkForwardWindow:
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


class WalkForwardRunner:
    def __init__(self, is_days: int = 365, oos_days: int = 90, step_days: int = 90):
        self.is_days = is_days
        self.oos_days = oos_days
        self.step_days = step_days

    def get_windows(self, start: str, end: str) -> list[WalkForwardWindow]:
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        windows = []
        cur = start
        while cur + pd.Timedelta(days=self.is_days + self.oos_days) <= end:
            is_end = cur + pd.Timedelta(days=self.is_days)
            oos_end = is_end + pd.Timedelta(days=self.oos_days)
            windows.append(WalkForwardWindow(
                is_start=cur, is_end=is_end,
                oos_start=is_end, oos_end=oos_end,
            ))
            cur += pd.Timedelta(days=self.step_days)
        return windows
