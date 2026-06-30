def is_oos_gap_alert(is_sharpe: float, oos_sharpe: float, threshold: float = 0.50) -> tuple[bool, float]:
    if is_sharpe == 0:
        return (abs(oos_sharpe) > threshold, float("inf"))
    relative_gap = abs(is_sharpe - oos_sharpe) / abs(is_sharpe)
    return (relative_gap > threshold, float(relative_gap))
