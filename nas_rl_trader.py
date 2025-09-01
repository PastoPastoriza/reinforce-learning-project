import argparse
import os
import pickle
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # Allow plotting without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import deque
import random
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

    metric: str
    hold_bonus_bps: float
    epsilon_min: float
    epsilon_decay: float
    epsilon: float
    lr: float
    l2: float
    reward: str
    patience: int
    trade_penalty_bps: float
    holding_cost_bps: float




# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def maybe_make_dir(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)



def compute_metric_from_portfolio_value(pv: np.ndarray, method: str = "end_value") -> float:
    """Compute validation metric from portfolio value series.

    Parameters
    ----------
    pv : np.ndarray
        Array of portfolio values over time.
    method : str
        Metric to compute: ``end_value`` | ``sharpe`` | ``pnl_dd``.
        ``pnl_dd`` scales the final value by ``1 / (1 + max_drawdown)``
        to reward profits while moderating large drawdowns.
    """
    if pv.size == 0:
        return float("-inf")
    if method == "end_value":
        return float(pv[-1])
    if method == "sharpe":
        r = np.diff(pv) / pv[:-1]
        if r.size == 0:
            return -1e9
        mu = np.mean(r)
        sigma = np.std(r) + 1e-12
        return float(mu / sigma * np.sqrt(24252))  # annualized Sharpe
    if method == "pnl_dd":
        end = float(pv[-1])
        peak = np.maximum.accumulate(pv)
        dd = np.max((peak - pv) / peak) if pv.size > 0 else 1.0
        # Penalize drawdowns by scaling the final value.
        # Metric grows when end-of-period PnL outweighs drawdown.
        return float(end / (1.0 + dd))
    return float(pv[-1])


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
        window_size: int = 5,
        fee_bps: float = 1.0,
        initial_capital: float = 10000.0,
        integer_shares: bool = False,

        hold_bonus_bps: float = 0.0,
        reward_mode: str = "return",
        trade_penalty_bps: float = 0.0,
        holding_cost_bps: float = 0.0,

    ):
        self.features = features
        self.bid_close = bid_close
        self.ask_close = ask_close
        self.timestamps = timestamps
        self.window_size = window_size
        self.fee = fee_bps / 10000.0
        self.initial_capital = initial_capital
        self.integer_shares = integer_shares
        self.reward_mode = reward_mode

        self.hold_bonus_bps = hold_bonus_bps / 10000.0
        self.trade_penalty = trade_penalty_bps / 10000.0
        self.holding_cost = holding_cost_bps / 10000.0

        self.n_features = features.shape[1]
        self.state_dim = window_size * self.n_features + 2  # + position, cash_norm

        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.cash = self.initial_capital
        self.shares = 0.0
        self.trades = []  # store logs for analysis
        state = self._get_state()
        self.portfolio_value = self.cash  # initial value
        return state

    def _get_state(self):
        start = self.current_step - self.window_size
        window = self.features[start : self.current_step]
        state = window.flatten()
        # Extras give agent context about holding status and liquidity
        if self.current_step == 0:
            bid = self.bid_close[0]
        else:
            bid = self.bid_close[self.current_step - 1]
        position_value = self.shares * bid
        total_value = self.cash + position_value
        position_frac = 0.0 if total_value == 0 else position_value / total_value
        cash_norm = self.cash / self.initial_capital
        return np.concatenate([state, [position_frac, cash_norm]])

    def step(self, action: int):
        bid = self.bid_close[self.current_step]
        ask = self.ask_close[self.current_step]
        timestamp = self.timestamps[self.current_step]

        prev_portfolio = self.cash + self.shares * bid

        current_frac = 0.0 if prev_portfolio == 0 else (self.shares * bid) / prev_portfolio
        target_frac = current_frac
        if action == 1:  # BUY -> go full long
            target_frac = 1.0
        elif action == 2:  # SELL -> exit to cash
            target_frac = 0.0

        target_value = prev_portfolio * target_frac
        target_shares = target_value / ask
        if self.integer_shares:
            target_shares = np.floor(target_shares)
        delta_shares = target_shares - self.shares

        exec_price = np.nan
        fee_paid = 0.0
        if delta_shares > 0:  # buy
            cost = delta_shares * ask
            fee_paid = cost * self.fee
            self.cash -= cost + fee_paid
            self.shares += delta_shares
            exec_price = ask
        elif delta_shares < 0:  # sell
            gross = -delta_shares * bid
            fee_paid = gross * self.fee
            self.cash += gross - fee_paid
            self.shares += delta_shares
            exec_price = bid

        self.portfolio_value = self.cash + self.shares * bid

        if self.reward_mode == "delta":
            reward = self.portfolio_value - prev_portfolio
        else:
            reward = (self.portfolio_value - prev_portfolio) / max(prev_portfolio, 1e-9)
        if (
            self.shares > 0
            and self.current_step > 0
            and self.bid_close[self.current_step] > self.bid_close[self.current_step - 1]
            and self.hold_bonus_bps > 0
        ):
            reward += self.hold_bonus_bps * prev_portfolio

        if delta_shares != 0 and self.trade_penalty > 0:
            reward -= self.trade_penalty * abs(delta_shares) * ask

        if self.shares > 0 and self.holding_cost > 0:
            reward -= self.holding_cost * prev_portfolio

        trade_info = {
            "timestamp": timestamp,
            "action": {0: "HOLD", 1: "BUY", 2: "SELL"}[action],

            "exec_price": exec_price,
            "position_after": target_frac,
            "shares_traded": delta_shares,
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

    def sgd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lr: float = 0.01,
        momentum: float = 0.9,
        l2: float = 0.0,
    ):
        assert X.ndim == 2
        num_values = np.prod(Y.shape)
        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values  # d/dW 1/N||Yhat-Y||^2
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values  # d/db 1/N||Yhat-Y||^2
        gW += l2 * self.W
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
        l2: float = 0.0,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update: int = 100,
        seed: int = 0,
        total_episodes: int = 1,
    ):
        np.random.seed(seed)
        random.seed(seed)
        self.n_action = n_action
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_min
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.total_episodes = total_episodes
        self.lr = lr
        self.l2 = l2
        self.batch_size = batch_size
        self.target_update = target_update
        self.buffer: deque = deque(maxlen=buffer_size)
        self.model = LinearModel(state_dim, n_action)
        self.target_model = LinearModel(state_dim, n_action)
        self.update_target()
        self.train_steps = 0

    def update_target(self):
        self.target_model.W = self.model.W.copy()
        self.target_model.b = self.model.b.copy()

    def act(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_action)
        q = self.model.predict(state.reshape(1, -1))[0]
        return int(np.argmax(q))

    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        states = np.vstack([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])
        next_states = np.vstack([b[3] for b in batch])
        dones = np.array([b[4] for b in batch]).astype(float)

        target = self.model.predict(states)
        q_next = self.target_model.predict(next_states)
        max_next = np.max(q_next, axis=1)
        target[np.arange(self.batch_size), actions] = rewards + self.gamma * (1 - dones) * max_next
        self.model.sgd(states, target, lr=self.lr, l2=self.l2)

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.update_target()

    def decay_epsilon(self, ep: int):
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start
            - (self.epsilon_start - self.epsilon_end) * ep / self.total_episodes,
        )

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

        hold_bonus_bps=args.hold_bonus_bps,
        reward_mode=args.reward,
        trade_penalty_bps=args.trade_penalty_bps,
        holding_cost_bps=args.holding_cost_bps,
    )


def play_one_episode(agent: LinearQAgent, env: SingleAssetEnv, is_train: bool):
    """Run a single episode and return end value and portfolio path."""
    state = env.reset()
    done = False
    pv = []
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        if is_train:
            agent.remember(state, action, reward, next_state, done)
            agent.train()
        state = next_state
        pv.append(info["portfolio_value"])
    return pv[-1], np.array(pv)


def temporal_split(df: pd.DataFrame, features: pd.DataFrame):
    train_mask = (df.index >= "2022-01-01") & (df.index <= "2023-12-31")
    val_mask = (df.index >= "2024-01-01") & (df.index <= "2024-12-31")
    test_mask = df.index >= "2025-01-01"
    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask]
    test_df = df.loc[test_mask]
    train_feat = features.loc[train_mask]
    val_feat = features.loc[val_mask]
    test_feat = features.loc[test_mask]

    print(
        f"Train -> rows: {len(train_df)}, range: {train_df.index.min()} to {train_df.index.max()}"
    )
    print(
        f"Val   -> rows: {len(val_df)}, range: {val_df.index.min()} to {val_df.index.max()}"
    )
    print(
        f"Test  -> rows: {len(test_df)}, range: {test_df.index.min()} to {test_df.index.max()}"
    )
    return train_df, val_df, test_df, train_feat, val_feat, test_feat


def run_train(df: pd.DataFrame, features: pd.DataFrame, args: Args):
    train_df, val_df, _, train_feat, val_feat, _ = temporal_split(df, features)


    scaler = StandardScaler()
    scaler.fit(train_feat.values)
    train_scaled = np.clip(scaler.transform(train_feat.values), -5, 5)

    val_scaled = np.clip(scaler.transform(val_feat.values), -5, 5)

    env = build_env(train_df, train_scaled, args)
    val_env = build_env(val_df, val_scaled, args)
    agent = LinearQAgent(
        env.state_dim,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        total_episodes=args.episodes,
    )

    os.makedirs("nas_rl_models", exist_ok=True)
    maybe_make_dir("nas_rl_rewards")
    best_val_score = -float("inf")
    best_weights_path = os.path.join("nas_rl_models", "weights_best.npz")
    last_weights_path = os.path.join("nas_rl_models", "weights_last.npz")

    train_end_values = []
    no_improve = 0
    for ep in range(args.episodes):
        end_val, _ = play_one_episode(agent, env, is_train=True)
        train_end_values.append(end_val)

        epsilon_actual = agent.epsilon
        agent.epsilon = 0.0
        _, pv_val = play_one_episode(agent, val_env, is_train=False)
        agent.epsilon = epsilon_actual
        metric = compute_metric_from_portfolio_value(pv_val, args.metric)
        
        if (ep + 1) % 10 == 0:
            print(
                f"episode: {ep + 1}/{args.episodes}, eps: {agent.epsilon:.4f}, train_end: {end_val:.2f}, val_{args.metric}: {metric:.2f}"
            )
        if metric > best_val_score:
            best_val_score = metric
            agent.save(best_weights_path)
            print(
                f"[VAL] new best checkpoint -> {best_val_score:.2f} saved at {best_weights_path}"
            )
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping due to no improvement")
                break

        agent.decay_epsilon(ep)

    agent.save(last_weights_path)
    with open("nas_rl_models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    np.save("nas_rl_rewards/train.npy", np.array(train_end_values))


def run_test(df: pd.DataFrame, features: pd.DataFrame, args: Args):
    _, _, test_df, _, _, test_feat = temporal_split(df, features)


    with open("nas_rl_models/scaler.pkl", "rb") as f:
        scaler: StandardScaler = pickle.load(f)
    test_scaled = np.clip(scaler.transform(test_feat.values), -5, 5)

    env = build_env(test_df, test_scaled, args)
    agent = LinearQAgent(
        env.state_dim,
        epsilon=0.0,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
        total_episodes=args.episodes,
    )


    best_path = os.path.join("nas_rl_models", "weights_best.npz")
    last_path = os.path.join("nas_rl_models", "weights_last.npz")
    if os.path.exists(best_path):
        agent.load(best_path)
        print(f"Loaded best weights from {best_path}")
    elif os.path.exists(last_path):
        agent.load(last_path)
        print(f"[WARN] best weights not found, using last weights from {last_path}")
    else:
        raise FileNotFoundError("No model weights found")
    agent.epsilon = 0.0


    maybe_make_dir("nas_rl_rewards")
    maybe_make_dir("nas_rl_trades")


    end_val, pv = play_one_episode(agent, env, is_train=False)
    np.save("nas_rl_rewards/test.npy", pv)

    trades_df = pd.DataFrame(env.trades)
    trades_df[trades_df["shares_traded"] != 0].to_csv(
        "nas_rl_trades/test_trades.csv", index=False
    )
    trades_df.to_csv("nas_rl_trades/test_actions_full.csv", index=False)

    dates = test_df.index[args.window - 1 : args.window - 1 + len(pv)]
    nasdaq_prices = test_df["BidClose"].iloc[
        args.window - 1 : args.window - 1 + len(pv)
    ]
    nas_ret = nasdaq_prices / nasdaq_prices.iloc[0] - 1
    strat_ret = pv / env.initial_capital - 1

    plt.figure(figsize=(10, 5))
    plt.plot(dates, nas_ret, label="Nasdaq")
    plt.plot(dates, strat_ret, label="Strategy")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.tight_layout()
    plt.savefig("nas_rl_trades/test_returns.png")


    print(f"episode end value: {end_val:.2f}")



def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", required=True, choices=["train", "test"])
    parser.add_argument("--csv", default="NAS100-m1.csv")
    parser.add_argument("--atr", type=int, default=14)
    parser.add_argument("--sma", type=int, default=40)
    parser.add_argument("--fee_bps", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=2000)

    parser.add_argument("--metric", choices=["end_value", "sharpe", "pnl_dd"], default="pnl_dd")
    parser.add_argument("--hold_bonus_bps", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_min", type=float, default=0.01)
    parser.add_argument("--epsilon_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--reward", choices=["delta", "return"], default="return")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--trade_penalty_bps", type=float, default=10.0)
    parser.add_argument("--holding_cost_bps", type=float, default=0.0)

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
