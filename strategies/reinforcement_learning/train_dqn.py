#agent will learn from intersection with environment by using a neural network to approximate Q-value function 
#Q-value function: expected future reward agent will get if it takes action in given state
import pandas as pd
from strategies.reinforcement_learning.rl_trading import StockTradingEnv
from strategies.reinforcement_learning.train_dqn import DQNAgent
import torch

if __name__ == "__main__":
    df = pd.read_csv("../data/NVDA_data.csv")
    
    env = StockTradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(input_dim=state_size, action_size=action_size)
    
    episodes = 1000
    batch_size = 32 #these are training params
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for time in range(env.max_steps):
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            agent.replay(batch_size)
            
            state = next_state
            total_reward += reward
            
            if done: 
                print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")
                break
        
        if(e + 1) % 50 == 0:
            agent.save(f"dqn_model_{e+1}.pth")
            
#DQN (deep-q-network) is algorithm in reinforcement learning that uses deep neural networks to approximate Q-values 
#q-value is long-term reward agent can receive by taking action in specific scenario 