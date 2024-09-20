#Custom environment that simulates stock market where RL agent can can interact
#here we'll define the trading environment where agent takes actions like buy sell and hold
import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    #custom OpenAI gym environment for trading stocks with reinforcement learning
    #RL is ML technique where we teach software to make decisions by mimicking human process of trial and error

    def __init__(self, df, initial_balance=10000):
        super(StockTradingEnv, self).__init__()
        
        self.df = df #stock price data
        self.initial_balance = initial_balance #initial balance 
        self.current_step = 0 
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance #portfolio net worth
        self.max_steps = len(df) #total trading days
        
        self.action_space = spaces.Discrete(3) #action space is buy, sell or hold
        
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(3,), dtype=np.float32) #observation space which includes stock prices, balance, shares held
    
    def reset(self):
        #for resetting environment to initial state
        self.balance = self.initial_balance
        self.shares_held = 0
        self.shares_held = self.initial_balance
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        #get stock observation
        stock_price = self.df['Adj Close'].iloc[self.current_step]
        return np.array([stock_price, self.balance, self.shares_held])
    
    def step(self, action):
        #execute one time step within environment
        #Action: 0 = hold, 1 = buy, 2 = sell
        
        current_price = self.df['Adj Close'].iloc[self.current_step]
        
        if action == 1: #buy
            shares_bought = self.balance // current_price #buy as many shares as possible
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action == 2: #sell
            self.balance += self.shares_held * current_price
            self.shares_held = 0
        
        self.net_worth = self.balance + self.shares_held * current_price #updated net worth
        
        reward = (self.net_worth - self.initial_balance) / self.initial_balance #reward (net worth change)
        
        self.current_step += 1 #move to next step
        
        done = self.current_step >= self.max_steps - 1 #check if done (end of data)
        
        return self._next_observation(), reward, done, {}
    
    def render(self, mode='human'):
        #render current state
        
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares Held: {self.shares_held}')
        print(f'Net Worth: {self.net_worth}')
        