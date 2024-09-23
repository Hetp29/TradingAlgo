#Custom environment that simulates stock market where RL agent can can interact
#here we'll define the trading environment where agent takes actions like buy sell and hold
import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    #custom OpenAI gym environment for trading stocks with reinforcement learning
    #RL is ML technique where we teach software to make decisions by mimicking human process of trial and error

    def __init__(self, df, initial_balance=10000, transaction_fee_percent=0.001):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(df)
        self.transaction_fee_percent = transaction_fee_percent
        
        # Action space: 0 = hold, 1 = buy, 2 = sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [current price, balance, shares held, net worth]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        stock_price = self.df['Adj Close'].iloc[self.current_step]
        return np.array([stock_price, self.balance, self.shares_held, self.net_worth])

    def step(self, action):
        current_price = self.df['Adj Close'].iloc[self.current_step]
        self._take_action(action, current_price)

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self._calculate_reward()

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1

        return self._next_observation(), reward, done, {}

    def _take_action(self, action, current_price):
        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            cost = shares_bought * current_price * (1 + self.transaction_fee_percent)
            self.balance -= cost
            self.shares_held += shares_bought
        elif action == 2:  # Sell
            revenue = self.shares_held * current_price * (1 - self.transaction_fee_percent)
            self.balance += revenue
            self.shares_held = 0

    def _calculate_reward(self):
        reward = (self.net_worth - self.initial_balance) / self.initial_balance
        return reward

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Shares Held: {self.shares_held}, Net Worth: {self.net_worth}')
