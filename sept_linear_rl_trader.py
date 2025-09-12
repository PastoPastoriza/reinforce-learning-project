import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
import itertools
import argparse
import re
import os
import pickle
import sys

from sklearn.preprocessing import StandardScaler


def load_and_process(csv_path = "/Users/jeronimopastoriza/Documents/Reinforce Learning Project/NAS100-m1.csv", atr_period = 14, sma_period = 50):
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





def get_scaler(env):
  # return scikit-learn scaler object to scale the states
  # Note: you could also populate the replay buffer here

  states = []
  for _ in range(env.n_step):
    action = np.random.choice(env.action_space)
    state, reward, done, info = env.step(action)
    states.append(state)
    if done:
      break

  scaler = StandardScaler()
  scaler.fit(states)
  return scaler




def maybe_make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)



class LinearModel:
  """ A linear regression model """
  def __init__(self, input_dim, n_action):
    self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
    self.b = np.zeros(n_action)

    # momentum terms
    self.vW = 0
    self.vb = 0

    self.losses = []

  def predict(self, X):
    # make sure X is N x D
    assert(len(X.shape) == 2)
    return X.dot(self.W) + self.b

  def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
    # make sure X is N x D
    assert(len(X.shape) == 2)

    # the loss values are 2-D
    # normally we would divide by N only
    # but now we divide by N x K
    num_values = np.prod(Y.shape)

    # do one step of gradient descent
    # we multiply by 2 to get the exact gradient
    # (not adjusting the learning rate)
    # i.e. d/dx (x^2) --> 2x
    Yhat = self.predict(X)
    gW = 2 * X.T.dot(Yhat - Y) / num_values
    gb = 2 * (Yhat - Y).sum(axis=0) / num_values

    # update momentum terms
    self.vW = momentum * self.vW - learning_rate * gW
    self.vb = momentum * self.vb - learning_rate * gb

    # update params
    self.W += self.vW
    self.b += self.vb

    mse = np.mean((Yhat - Y)**2)
    self.losses.append(mse)

  def load_weights(self, filepath):
    npz = np.load(filepath)
    self.W = npz['W']
    self.b = npz['b']

  def save_weights(self, filepath):
    np.savez(filepath, W=self.W, b=self.b)




class MultiStockEnv:
  """
  A 3-stock trading environment.
  State: vector of size 7 (n_stock * 2 + 1)
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - # shares of stock 3 owned
    - price of stock 1 (using daily close price)
    - price of stock 2
    - price of stock 3
    - cash owned (can be used to purchase more stocks)
  Action: categorical variable with 27 (3^3) possibilities
    - for each stock, you can:
    - 0 = sell
    - 1 = hold
    - 2 = buy
  """
  def __init__(self, data, initial_investment=20000):
    # data
    self.stock_price_history = data
    self.n_step, self.n_features = self.stock_price_history.shape
    self.n_stock = 1  # we only have 1 stock in this env

    # instance attributes
    self.initial_investment = initial_investment
    self.cur_step = None
    self.stock_owned = None
    self.stock_price = None
    # self.cash_in_hand = None
    self.balance = np.nan
    self.in_trade = False
    self.stock_return = None

    self.action_space = np.arange(3**self.n_stock)

    # action permutations
    # returns a nested list with elements like:
    # [0,0,0]
    # [0,0,1]
    # [0,0,2]
    # [0,1,0]
    # [0,1,1]
    # etc.
    # 0 = sell
    # 1 = hold
    # 2 = buy
    self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))

    # calculate size of state
    # self.state_dim = self.n_features # x window
    self.balance_state_dim = self.n_stock * 2 + 1 + 1 # +1 for balance, +1 for in_trade status
    self.state_dim = self.balance_state_dim

    self.reset()


  def reset(self):
    self.cur_step = 0
    self.stock_owned = np.zeros(self.n_stock)
    self.stock_price = 0 #self.stock_price_history.iloc[self.cur_step]
    # self.cash_in_hand = self.initial_investment
    self.balance = self.initial_investment
    self.stock_return = 0 
    return self._get_obs()


  def step(self, action):
    assert action in self.action_space

    # get current value before performing the action
    # prev_val = self._get_val()
    prev_val = self.balance

    # update price, i.e. go to the next day
    self.cur_step += 1
    # self.stock_price = self.stock_price_history.iloc[self.cur_step]
    self.stock_return = self.stock_price_history["log_return_close"].iloc[self.cur_step]

    # perform the trade
    self._trade(action)

    # get the new value after taking the action
    cur_val = self.balance

    # reward is the increase in porfolio value
    reward = cur_val - prev_val
    # reward = self.stock_price_history["log_return_close"].iloc[self.cur_step]

    # done if we have run out of data
    done = self.cur_step == self.n_step - 1

    # store the current value of the portfolio here
    info = {'cur_val': cur_val}

    # conform to the Gym API
    return self._get_obs(), reward, done, info


  def _get_obs(self):
    obs = np.empty(self.balance_state_dim)
    obs[:self.n_stock] = self.stock_owned
    # obs[self.n_stock:2*self.n_stock] = self.stock_price
    obs[self.n_stock:2*self.n_stock] = self.stock_return
    # obs[-1] = self.cash_in_hand
    obs[-2] = self.balance
    obs[-1] = self.in_trade
    return obs
    


  # def _get_val(self):
  #   # return self.stock_owned.dot(self.stock_price) + self.cash_in_hand
  #   self.balance = self.balance * self.stock_return
  #   return self.balance


  def _trade(self, action):
    # index the action we want to perform
    # 0 = sell
    # 1 = hold
    # 2 = buy
    # e.g. [2,1,0] means:
    # buy first stock
    # hold second stock
    # sell third stock
    action_vec = self.action_list[action]

    # determine which stocks to buy or sell
    sell_index = [] # stores index of stocks we want to sell
    buy_index = [] # stores index of stocks we want to buy
    hold_index = [] # stores index of stocks we want to hold
    for i, a in enumerate(action_vec):
      if a == 0:
        sell_index.append(i)
      elif a == 2:
        buy_index.append(i)
      elif a == 1:
        hold_index.append(i)

    # sell any stocks we want to sell
    # then buy any stocks we want to buy
    if sell_index and self.in_trade:
      # NOTE: to simplify the problem, when we sell, we will sell ALL shares of that stock
      for i in sell_index:
        # self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
        self.balance *= np.exp(self.stock_return)

        #VOY A DEJAR ACA. ESTABA CAMBIANDO STOCK_PRICE POR STOCK_RETURN. PERO AL CALCULAR AL VENDER, NO TIENE SENTIDO. 
        #TENGO QUE CALCULAR AL VENDER POR EL PRECIO, O POR VELA CON EL RETURN.
        self.stock_owned[i] = 0
        self.in_trade = False
    if buy_index and (not self.in_trade):
      # NOTE: when buying, we will loop through each stock we want to buy,
      #       and buy one share at a time until we run out of cash
      # can_buy = True
      # while can_buy:
      for i in buy_index:
        # if self.cash_in_hand > self.stock_price[i]:
        # self.stock_owned[i] = 1 # buy one share
        # self.cash_in_hand -= self.stock_price[i]
        # self.balance = self.balance * self.stock_return[i] porque si compro en close, no tiene sentido actualizar el balance con el return
        # else:
        #   can_buy = False
        self.in_trade = True
    if hold_index:
      for i in hold_index:
        if self.in_trade:
          self.balance *= np.exp(self.stock_return)
          self.stock_owned[i] = 1
        else:
          self.balance = self.balance
          self.stock_owned[i] = 0






class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.action_size = action_size
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    # self.epsilon_decay = 0.995
    self.epsilon_decay = (self.epsilon_min / self.epsilon) ** (1 / n_timesteps*2)
    self.model = LinearModel(state_size, action_size)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)
    act_values = self.model.predict(state)
    return np.argmax(act_values[0])  # returns action


  def train(self, state, action, reward, next_state, done):
    if done:
      # target = reward
      target = float(reward)
    else:
      # target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)
      q_next = self.model.predict(next_state)[0]  # (n_action,)
      target = float(reward + self.gamma * np.max(q_next))


    target_full = self.model.predict(state)
    target_full[0, action] = target

    # Run one training step
    self.model.sgd(state, target_full, learning_rate=1e-4)

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay


  def load(self, name):
    self.model.load_weights(name)


  def save(self, name):
    self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
  # note: after transforming states are already 1xD
  state = env.reset()
  state = scaler.transform([state])
  done = False


  timestamps = []
  actions = []
  pvs = []


  while not done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)
    ts = env.stock_price_history.index[env.cur_step]
    timestamps.append(ts)
    actions.append(action)
    pvs.append(float(info['cur_val']))
    next_state = scaler.transform([next_state])
    if is_train == 'train':
      agent.train(state, action, reward, next_state, done)
    state = next_state

  # return info['cur_val']
  if env.in_trade:
    timestamps.append(env.stock_price_history.index[env.cur_step])
    actions.append(0) # sell
    pvs.append(float(info['cur_val']))

  return pvs[-1], {'timestamps': timestamps, 'actions': actions, 'pvs': pvs}





def plot_and_export_test(history, env, out_plot='test_returns.png', out_csv='test_trades.csv'):

# Buy-and-hold cumulative return from log returns (aligned to logged timestamps)
  idx = pd.Index(history['timestamps'])
  ret = env.stock_price_history['log_return_close'].reindex(idx).astype(float).values
  # bh_curve = np.exp(np.cumsum(ret))
  # if len(bh_curve) > 0:
  #   bh_curve = bh_curve / bh_curve[0]
  bh_ret_pct = (np.exp(np.cumsum(ret)) - 1.0) * 100.0

  # Strategy PV normalized to 1
  pvs = np.array(history['pvs'], dtype=float)
  # strat_curve = pvs / pvs[0] if len(pvs) > 0 and pvs[0] != 0 else pvs
  strat_ret_pct = (pvs / env.initial_investment - 1.0) * 100.0

  # Human-readable actions
  action_names = ['sell' if a == 0 else 'hold' if a == 1 else 'buy' for a in history['actions']]

  # CSV with all steps (filter later to just buys/sells if you prefer)
  trades_df = pd.DataFrame({
  'timestamp': history['timestamps'],
  'action': history['actions'],
  'action_name': action_names,
  'portfolio_value': pvs,
  # 'strategy_cumret': strat_curve,
  # 'bh_cumret': bh_curve,
  'strategy_ret_pct': strat_ret_pct,
  'bh_ret_pct': bh_ret_pct,
  })
  trades_df = trades_df[trades_df['action'] != 1].copy() # filter only buys/sells
  trades_df.to_csv(out_csv, index=False)
  print(f'Wrote trades CSV: {out_csv}')

  # Plot and save
  plt.figure(figsize=(10, 5))
  plt.plot(history['timestamps'], strat_ret_pct, label='Strategy return (%)')
  plt.plot(history['timestamps'], bh_ret_pct, label='Buy & Hold return (%)')
  plt.title('Strategy vs Buy & Hold (Test)')
  plt.xlabel('Time')
  plt.ylabel('Return (%)')
  plt.grid(True, alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_plot, dpi=150)
  print(f'Saved plot: {out_plot}')
  try:
    plt.show()
  except Exception:
    pass








if __name__ == '__main__':

  # config
  models_folder = 'sep_linear_rl_trader_models'
  rewards_folder = 'sep_linear_rl_trader_rewards'
  num_episodes = 2000
  batch_size = 32
  initial_investment = 20000


  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--mode', type=str, required=True,
                      help='either "train" or "test"')
  args = parser.parse_args()

  maybe_make_dir(models_folder)
  maybe_make_dir(rewards_folder)

  data, features = load_and_process()
  n_timesteps, N_features = features.shape
  # n_stocks = 1  # we only have 1 stock in this env

  n_train = n_timesteps // 2

  train_data = features[:n_train]
  test_data = features[n_train:]

  env = MultiStockEnv(train_data, initial_investment)
  state_size = env.state_dim
  action_size = len(env.action_space)
  agent = DQNAgent(state_size, action_size)
  scaler = get_scaler(env)

  # store the final value of the portfolio (end of episode)
  portfolio_value = []

  if args.mode == 'test':
    # then load the previous scaler #
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)

    # remake the env with test data
    env = MultiStockEnv(test_data, initial_investment)

    # make sure epsilon is not 1!
    # no need to run multiple episodes if epsilon = 0, it's deterministic
    agent.epsilon = 0

    # load trained weights
    agent.load(f'{models_folder}/linear.npz')

  # play the game num_episodes times
  if args.mode == 'train':
    for e in range(num_episodes):
      # t0 = datetime.now()
      end_val, _ = play_one_episode(agent, env, args.mode)
      # dt = datetime.now() - t0
      if e % 100 == 0:
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {end_val:.2f}, epsilon: {agent.epsilon}")
      portfolio_value.append(end_val) # append episode end portfolio value
  else:
    # val = play_one_episode(agent, env, args.mode)
    # print(f"test episode end value: {val:.2f}")
    # portfolio_value.append(val) # append episode end portfolio value
    num_episodes = 1

    end_val, test_history = play_one_episode(agent, env, args.mode)

    plot_and_export_test(
    test_history,
    env,
    out_plot=f'{rewards_folder}/test_returns.png',
    out_csv=f'{rewards_folder}/test_trades.csv'
    )

    portfolio_value.append(end_val)
    print(f'Test episode end value: {end_val:.2f}')
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
    sys.exit(0)





  # save the weights when we are done
  if args.mode == 'train':
    # save the DQN
    agent.save(f'{models_folder}/linear.npz')

    # save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
      pickle.dump(scaler, f)

    # plot losses
    plt.plot(agent.model.losses)
    plt.show()


  # save portfolio value for each episode
  np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)
