def load_and_process(csv_path, atr_period, sma_period):
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
    hist_max = close.shift(1).rolling(window=1000000, min_periods=1).max()
    hist_min = close.shift(1).rolling(window=1000000, min_periods=1).min()
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
