import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_panel(
    n_tickers: int = 50,
    n_days: int = 365,
    n_factors: int = 16,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D", tz="UTC")
    tickers = [f"TICKER_{i:03d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date_utc", "ticker"])

    data = {}
    for i in range(n_factors):
        data[f"factor_{i:02d}"] = rng.normal(0, 1, len(idx))

    data["close"] = 100 * np.exp(rng.normal(0, 0.02, len(idx)).cumsum())
    data["volume"] = rng.uniform(1e6, 1e8, len(idx))
    data["market_cap"] = rng.uniform(1e8, 1e10, len(idx))
    data["funding_rate"] = rng.normal(0, 0.001, len(idx))
    taker_buy_base = rng.uniform(1e6, 1e8, len(idx))
    taker_sell_base = rng.uniform(1e6, 1e8, len(idx))
    taker_buy_ratio = taker_buy_base / (taker_buy_base + taker_sell_base)
    taker_net_volume = (taker_buy_base - taker_sell_base) / (taker_buy_base + taker_sell_base)
    data["taker_buy_ratio"] = taker_buy_ratio
    data["taker_net_volume"] = taker_net_volume
    data["oi_usd"] = rng.uniform(1e7, 1e9, len(idx))
    data["ls_ratio"] = rng.uniform(0.3, 0.7, len(idx))
    categories = ["DeFi", "L1", "L2", "Meme", "Gaming"]
    data["category"] = rng.choice(categories, len(idx))

    df = pd.DataFrame(data, index=idx)
    return df


def main():
    output_dir = Path(__file__).parent.parent / "tests" / "fixtures"
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = generate_synthetic_panel()
    path = output_dir / "synthetic_panel.pkl"
    panel.to_pickle(path)
    print(f"Generated {path} ({len(panel)} rows)")


if __name__ == "__main__":
    main()
