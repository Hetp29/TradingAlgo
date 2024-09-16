#Custom environment that simulates stock market where RL agent can can interact
#here we'll define the trading environment where agent takes actions like buy sell and hold
import numpy as np
import pandas as pd
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    #custom OpenAI gym environment for trading stocks with reinforcement learning
    #RL is ML technique where we teach software to make decisions by mimicking human process of trial and error
    
    