import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_and_process(csv_path: str, atr_period: int, sma_period: int):
    """Load NASDAQ data, resample to 1H and compute features.

    The feature set is intentionally narrow so the agent only observes
    information believed to be predictive.  The DataFrame returned keeps the
    original OHLC columns for inspection, but the ``features`` DataFrame only
    contains the engineered signals listed below.

    Parameters
    ----------
    csv_path : str
        Path to minute-level NASDAQ data with bid/ask OHLC and volume.
    atr_period : int
        Window size for Average True Range.
    sma_period : int
        Window size for Simple Moving Average.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Full DataFrame and feature subset used for the model.
    """
    # 1) Read CSV and adjust timestamps (UTC-3 -> UTC)
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Date"] = df["Date"] - pd.Timedelta(hours=3)
    df = df.sort_values("Date").set_index("Date")

    # 2) Resample to 1H per side with specified rules
    agg = {
        "BidOpen": "first",
        "BidHigh": "max",
        "BidLow": "min",
        "BidClose": "last",
        "AskOpen": "first",
        "AskHigh": "max",
        "AskLow": "min",
        "AskClose": "last",
        "Volume": "sum",
    }
    df = df.resample("1h").agg(agg).dropna()

    # 3) Indicators on BID prices
    high = df["BidHigh"]
    low = df["BidLow"]
    open_ = df["BidOpen"]
    close = df["BidClose"]

    # True range components -> ATR
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,  # high-low
            (high - prev_close).abs(),  # |high-prev_close|
            (prev_close - low).abs(),  # |prev_close-low|
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    sma = close.rolling(window=sma_period).mean()

    # --- Feature engineering -------------------------------------------------
    log_return_close = np.log(close / close.shift(1))  # ln(C_t / C_{t-1})
    log_return_high = np.log(high / high.shift(1))  # ln(H_t / H_{t-1})
    log_return_low = np.log(low / low.shift(1))  # ln(L_t / L_{t-1})
    candle_direction = (close > open_).astype(float)  # 1 if green else 0
    atr_pct = atr / close  # ATR_t / Close_t
    sma_gradient = sma.diff()  # SMA_t - SMA_{t-1}
    close_sma_pct = (close - sma) / sma  # (Close_t - SMA_t) / SMA_t
    volume_diff = df["Volume"].diff()  # Volume_t - Volume_{t-1}
    hist_max = close.cummax()
    hist_min = close.cummin()
    historical_max_pct = close / hist_max - 1  # Close_t / max_{<=t} - 1
    historical_min_pct = close / hist_min - 1  # Close_t / min_{<=t} - 1
    rel_max = close.shift(1).rolling(window=50, min_periods=1).max()
    rel_min = close.shift(1).rolling(window=50, min_periods=1).min()
    relative_max_pct = close / rel_max - 1  # Close_t / max_{t-50:t-1} - 1
    relative_min_pct = close / rel_min - 1  # Close_t / min_{t-50:t-1} - 1

    df["ATR"] = atr
    df["SMA"] = sma
    df["log_return_close"] = log_return_close
    df["log_return_high"] = log_return_high
    df["log_return_low"] = log_return_low
    df["candle_direction"] = candle_direction
    df["ATR_PCT"] = atr_pct
    df["SMA_gradient"] = sma_gradient
    df["close_sma_pct"] = close_sma_pct
    df["volume_diff"] = volume_diff
    df["historical_max_pct"] = historical_max_pct
    df["relative_max_pct"] = relative_max_pct
    df["historical_min_pct"] = historical_min_pct
    df["relative_min_pct"] = relative_min_pct

    # 4) Drop rows with NaN from indicators/returns (warmup period)
    df = df.dropna()

    feature_cols = [
        "log_return_close",
        "log_return_high",
        "log_return_low",
        "candle_direction",
        "ATR_PCT",
        "SMA_gradient",
        "close_sma_pct",
        "volume_diff",
        "historical_max_pct",
        "relative_max_pct",
        "historical_min_pct",
        "relative_min_pct",
    ]

    features = df[feature_cols]
    # volume_diff reflects changes in market activity without ratio noise

    # Logging shapes and ranges
    print(
        f"Data processed -> shape: {df.shape}, range: {df.index.min()} to {df.index.max()}"
    )

    return df, features

def train(args):
    df, features = load_and_process(args.csv, args.atr, args.sma)

    price_all = df["BidClose"].astype(float).values
    feat_all = features.values.astype(float)
    split = int(len(price_all) * 0.75)
    price = price_all[:split]
    feat = feat_all[:split]
    mean = feat.mean(axis=0)
    std = feat.std(axis=0) + 1e-8
    feat = (feat - mean) / std
    n = len(price)

    print(f"Train/test split -> train:{n}, test:{len(price_all) - n}")
    
    rng = np.random.default_rng(args.seed)
    d = feat.shape[1] + 1
    w = rng.normal(scale=0.001, size=(3, d))
    b = np.zeros(3)
    gamma = 0.99
    lr = 0.0001
    total_steps = max(1, (n - 1) * args.episodes)
    step = 0
    best_pv = -np.inf
    best_w, best_b = w.copy(), b.copy()
    wait = 0
    for ep in range(1, args.episodes + 1):
        cash = 1.0
        shares = 0.0
        pos = 0
        pv = 1.0
        peak = pv
        dd_prev = 0.0
        for t in range(1, n):
            s = np.concatenate([feat[t - 1], [float(pos)]])
            eps = max(0.05, 1.0 - step / (total_steps - 1) * 0.95)
            step += 1
            if rng.random() < eps:
                a = rng.integers(0, 3)
            else:
                a = int(np.argmax(w @ s + b))
            p = price[t - 1]
            if a == 1 and pos == 0:
                shares = (cash * (1 - args.tx_cost)) / p
                cash = 0.0
                pos = 1
            elif a == 2 and pos == 1:
                cash = shares * p * (1 - args.tx_cost)
                shares = 0.0
                pos = 0
            pv_mid = cash + shares * p
            p_next = price[t]
            pv_next = cash + shares * p_next
            strat_ret = pv_next / pv_mid - 1.0
            bench_ret = p_next / p - 1.0
            peak = max(peak, pv_next)
            dd = 1 - pv_next / peak
            dd_step = max(0.0, dd - dd_prev)
            reward = args.adv_weight * (strat_ret - bench_ret) - args.dd_penalty * dd_step
            s_next = np.concatenate([feat[t], [float(pos)]])
            q_sa = w[a] @ s + b[a]
            q_next = np.max(w @ s_next + b) if t < n - 1 else 0.0
            target = reward + gamma * q_next * (t < n - 1)
            grad = q_sa - target
            w[a] -= lr * grad * s
            b[a] -= lr * grad
            pv = pv_next
            dd_prev = dd
        print(f"Episode {ep} end value: {pv:.3f}, epsilon: {eps:.2f}")
        if pv > best_pv:
            best_pv = pv
            best_w = w.copy()
            best_b = b.copy()
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping")
                break
    np.savez(
        "nas_rl_model_2.npz",
        w=best_w,
        b=best_b,
        mean=mean,
        std=std,
        feature_cols=features.columns.values,
    )


def test(args):
    df, features = load_and_process(args.csv, args.atr, args.sma)
    price_all = df["BidClose"].astype(float).values
    feat_all = features.values.astype(float)
    data = np.load("nas_rl_model_2.npz")
    mean = data["mean"]
    std = data["std"]
    feat_all = (feat_all - mean) / std
    split = int(len(price_all) * 0.75)
    price = price_all[split:]
    feat = feat_all[split:]
    idx = df.index[split:]
    n = len(price)
    w = data["w"]
    b = data["b"]
    cash = 1.0
    shares = 0.0
    pos = 0
    trades = []
    strat_rets = []
    bench_rets = []
    for t in range(1, n):

        s = np.concatenate([feat[t - 1], [float(pos)]])
        a = int(np.argmax(w @ s + b))
        p = price[t - 1]
        traded = False
        if a == 1 and pos == 0:
            shares = (cash * (1 - args.tx_cost)) / p
            cash = 0.0
            pos = 1
            traded = True
            act = "BUY"
        elif a == 2 and pos == 1:
            cash = shares * p * (1 - args.tx_cost)
            shares = 0.0
            pos = 0
            traded = True
            act = "SELL"
        pv_mid = cash + shares * p
        p_next = price[t]
        pv = cash + shares * p_next
        strat_ret = pv / pv_mid - 1.0
        bench_ret = p_next / p - 1.0
        strat_rets.append(strat_ret)
        bench_rets.append(bench_ret)
        if traded:
            trades.append({
                "timestamp": idx[t - 1],

                "action": act,
                "price": p,
                "shares": shares,
                "cash": cash,
                "position_value": shares * p,
                "portfolio_value": pv_mid,
            })
    cols = [
        "timestamp",
        "action",
        "price",
        "shares",
        "cash",
        "position_value",
        "portfolio_value",
    ]
    pd.DataFrame(trades, columns=cols).to_csv("test_trades.csv", index=False)
    strat_cum = np.cumprod(1 + np.array(strat_rets)) - 1
    bench_cum = np.cumprod(1 + np.array(bench_rets)) - 1
    plt.figure(figsize=(8, 4))
    plt.plot(idx[1:], strat_cum, label="Strategy")
    plt.plot(idx[1:], bench_cum, label="Buy&Hold NASDAQ")
    plt.title("Strategy vs Buy&Hold NASDAQ")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_returns.png")
    pd.DataFrame(
        {
            "timestamp": idx[1:],
            "strategy": strat_cum,
            "benchmark": bench_cum,
        }
    ).to_csv("test_results.csv", index=False)
    final_pv = cash + shares * price[-1]
    print(f"Test end value: {final_pv:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("-m", "--mode", choices=["train", "test"], default="test")
    p.add_argument("--atr", type=int, default=14)
    p.add_argument("--sma", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tx_cost", type=float, default=0.0025)
    p.add_argument("--dd_penalty", type=float, default=1.0)
    p.add_argument("--adv_weight", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
