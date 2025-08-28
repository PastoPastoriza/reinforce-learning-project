import argparse
import os
import pickle
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Allow plotting without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Args:
    mode: str
    csv: str
    atr: int
    sma: int
    fee_bps: float
    seed: int
    window: int
    episodes: int


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def maybe_make_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_and_process(csv_path: str, atr_period: int, sma_period: int):
    """Load NASDAQ data, resample to 1H and compute features.

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
    pd.DataFrame
        DataFrame indexed by timestamp with BidClose/AskClose and features.
    """
    # 1) Read CSV and adjust timestamps
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
    close = df["BidClose"]

    # True range components
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (prev_close - low).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    sma = close.rolling(window=sma_period).mean()

    log_return = np.log(close / close.shift(1))

    df["ATR"] = atr
    df["SMA"] = sma
    df["log_return_close"] = log_return

    # 4) Drop rows with NaN from indicators / first log return
    df = df.dropna()

    features = df[["log_return_close", "ATR", "SMA", "Volume"]]
    # Volume alone captures market activity without adding noise of returns

    # Logging shapes and ranges
    print(
        f"Data processed -> shape: {df.shape}, range: {df.index.min()} to {df.index.max()}"
    )

    return df, features


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class SingleAssetEnv:
    """Single asset trading environment with windowed state."""

    def __init__(
        self,
        features: np.ndarray,
        bid_close: np.ndarray,
        ask_close: np.ndarray,
        timestamps: pd.Index,
        window_size: int = 10,
        fee_bps: float = 1.0,
        initial_capital: float = 10000.0,
        integer_shares: bool = False,
    ):
        self.features = features
        self.bid_close = bid_close
        self.ask_close = ask_close
        self.timestamps = timestamps
        self.window_size = window_size
        self.fee = fee_bps / 10000.0
        self.initial_capital = initial_capital
        self.integer_shares = integer_shares
        self.n_features = features.shape[1]
        self.state_dim = window_size * self.n_features + 2  # + position, cash_norm

        self.reset()

    def reset(self):
        self.current_step = self.window_size - 1
        self.cash = self.initial_capital
        self.shares = 0.0
        self.trades = []  # store logs for analysis
        state = self._get_state()
        self.portfolio_value = self.cash  # initial value
        return state

    def _get_state(self):
        start = self.current_step - self.window_size + 1
        window = self.features[start : self.current_step + 1]
        state = window.flatten()
        # Extras give agent context about holding status and liquidity
        position = 1.0 if self.shares > 0 else 0.0
        cash_norm = self.cash / self.initial_capital
        return np.concatenate([state, [position, cash_norm]])

    def step(self, action: int):
        bid = self.bid_close[self.current_step]
        ask = self.ask_close[self.current_step]
        timestamp = self.timestamps[self.current_step]

        position_before = 1 if self.shares > 0 else 0
        prev_portfolio = self.cash + self.shares * bid

        exec_price = np.nan
        shares_traded = 0.0
        fee_paid = 0.0

        if action == 2:  # Buy / all-in
            if self.cash > 0:
                if self.integer_shares:
                    shares_traded = np.floor(self.cash / ask)
                    cost = shares_traded * ask
                    fee_paid = cost * self.fee
                    self.cash -= cost + fee_paid
                else:
                    fee_paid = self.cash * self.fee
                    cash_to_spend = self.cash - fee_paid
                    shares_traded = cash_to_spend / ask
                    self.cash = 0.0
                self.shares += shares_traded
                exec_price = ask
        elif action == 0:  # Sell / close
            if self.shares > 0:
                gross = self.shares * bid
                fee_paid = gross * self.fee
                self.cash += gross - fee_paid
                shares_traded = -self.shares
                self.shares = 0.0
                exec_price = bid
        # else: hold -> nothing changes

        position_after = 1 if self.shares > 0 else 0
        self.portfolio_value = self.cash + self.shares * bid
        reward = self.portfolio_value - prev_portfolio

        trade_info = {
            "timestamp": timestamp,
            "action": {0: "SELL", 1: "HOLD", 2: "BUY"}[action],
            "exec_price": exec_price,
            "position_before": position_before,
            "position_after": position_after,
            "shares_traded": shares_traded,
            "fee_paid": fee_paid,
            "cash": self.cash,
            "portfolio_value": self.portfolio_value,
        }
        self.trades.append(trade_info)

        self.current_step += 1
        done = self.current_step >= len(self.features)
        next_state = self._get_state() if not done else np.zeros(self.state_dim)

        info = {"portfolio_value": self.portfolio_value}
        return next_state, reward, done, info


# -----------------------------------------------------------------------------
# Linear Q-learning agent
# -----------------------------------------------------------------------------


class LinearModel:
    def __init__(self, input_dim: int, n_action: int):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)
        self.vW = 0.0
        self.vb = 0.0
        self.losses = []

    def predict(self, X: np.ndarray):
        assert X.ndim == 2
        return X.dot(self.W) + self.b

    def sgd(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.01, momentum: float = 0.9):
        assert X.ndim == 2
        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values
        self.vW = momentum * self.vW - lr * gW
        self.vb = momentum * self.vb - lr * gb
        self.W += self.vW
        self.b += self.vb
        self.losses.append(np.mean((Yhat - Y) ** 2))

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, b=self.b)

    def load(self, path: str) -> None:
        npz = np.load(path)
        self.W = npz["W"]
        self.b = npz["b"]


class LinearQAgent:
    def __init__(
        self,
        state_dim: int,
        n_action: int = 3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        lr: float = 0.001,
        seed: int = 0,
    ):
        np.random.seed(seed)
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.model = LinearModel(state_dim, n_action)

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_action)
        q = self.model.predict(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def train(self, state, action, reward, next_state, done):
        state = state.reshape(1, -1)
        next_state = next_state.reshape(1, -1)
        target = self.model.predict(state)
        if done:
            target[0, action] = reward
        else:
            q_next = self.model.predict(next_state)[0]
            target[0, action] = reward + self.gamma * np.max(q_next)
        self.model.sgd(state, target, lr=self.lr)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model.load(path)


# -----------------------------------------------------------------------------
# Main training / testing routines
# -----------------------------------------------------------------------------


def build_env(df: pd.DataFrame, features: np.ndarray, args: Args) -> SingleAssetEnv:
    bid_close = df["BidClose"].values
    ask_close = df["AskClose"].values
    return SingleAssetEnv(
        features=features,
        bid_close=bid_close,
        ask_close=ask_close,
        timestamps=df.index,
        window_size=args.window,
        fee_bps=args.fee_bps,
    )


def temporal_split(df: pd.DataFrame, features: pd.DataFrame):
    train_mask = (df.index >= "2021-01-01") & (df.index <= "2023-12-31")
    test_mask = df.index >= "2024-01-01"
    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    train_feat = features.loc[train_mask]
    test_feat = features.loc[test_mask]

    print(
        f"Train -> shape: {train_df.shape}, range: {train_df.index.min()} to {train_df.index.max()}"
    )
    print(
        f"Test  -> shape: {test_df.shape}, range: {test_df.index.min()} to {test_df.index.max()}"
    )
    return train_df, test_df, train_feat, test_feat


def run_train(df: pd.DataFrame, features: pd.DataFrame, args: Args):
    train_df, _, train_feat, _ = temporal_split(df, features)

    scaler = StandardScaler()
    scaler.fit(train_feat.values)
    train_scaled = scaler.transform(train_feat.values)

    env = build_env(train_df, train_scaled, args)
    agent = LinearQAgent(env.state_dim, seed=args.seed)

    maybe_make_dir("nas_rl_models")
    maybe_make_dir("nas_rl_rewards")

    portfolio_values = []
    for ep in range(args.episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.train(state, action, reward, next_state, done)
            state = next_state
        portfolio_values.append(info["portfolio_value"])
        print(
            f"episode: {ep + 1}/{args.episodes}, episode end value: {info['portfolio_value']:.2f}"
        )

    agent.save("nas_rl_models/weights.npz")
    with open("nas_rl_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("nas_rl_rewards/train.npy", np.array(portfolio_values))


def run_test(df: pd.DataFrame, features: pd.DataFrame, args: Args):
    _, test_df, _, test_feat = temporal_split(df, features)

    with open("nas_rl_models/scaler.pkl", "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    test_scaled = scaler.transform(test_feat.values)

    env = build_env(test_df, test_scaled, args)
    agent = LinearQAgent(env.state_dim, seed=args.seed)
    agent.load("nas_rl_models/weights.npz")
    agent.epsilon = 0.0  # deterministic policy

    maybe_make_dir("nas_rl_rewards")
    maybe_make_dir("nas_rl_trades")

    state = env.reset()
    done = False
    portfolio_values = []
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        state = next_state
        portfolio_values.append(info["portfolio_value"])

    np.save("nas_rl_rewards/test.npy", np.array(portfolio_values))

    trades_df = pd.DataFrame(env.trades)
    trades_df.to_csv("nas_rl_trades/test_trades.csv", index=False)

    # Plot strategy vs index returns
    dates = test_df.index[args.window - 1 : args.window - 1 + len(portfolio_values)]
    nasdaq_prices = test_df["BidClose"].iloc[args.window - 1 : args.window - 1 + len(portfolio_values)]
    nasdaq_ret = nasdaq_prices / nasdaq_prices.iloc[0] - 1
    strat_ret = np.array(portfolio_values) / env.initial_capital - 1

    plt.figure(figsize=(10, 5))
    plt.plot(dates, nasdaq_ret, label="Nasdaq")
    plt.plot(dates, strat_ret, label="Strategy")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig("nas_rl_trades/test_returns.png")

    if portfolio_values:
        print(f"episode end value: {portfolio_values[-1]:.2f}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, choices=["train", "test"])
    parser.add_argument("--csv", default="NAS100-m1.csv")
    parser.add_argument("--atr", type=int, default=14)
    parser.add_argument("--sma", type=int, default=40)
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    return Args(**vars(args))


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)

    df, features = load_and_process(args.csv, args.atr, args.sma)

    if args.mode == "train":
        run_train(df, features, args)
    else:
        run_test(df, features, args)
