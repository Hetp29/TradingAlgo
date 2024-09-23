#agent will learn from intersection with environment by using a neural network to approximate Q-value function 
#Q-value function: expected future reward agent will get if it takes action in given state
import pandas as pd
from rl_trading import StockTradingEnv
from dqn_agent import DQNAgent
import torch
import numpy as np
import matplotlib.pyplot as plt


episodes = 1000
batch_size = 64
learning_rate = 0.001
discount_factor = 0.99
epsilon_start = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
target_update = 10  

def train_dqn(df, agent, env, episodes=1000, batch_size=64, target_update=10):
    total_rewards = []
    losses = []
    best_reward = -float('inf')
    best_model = None
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for time in range(env.max_steps):
            # Agent chooses action
            action = agent.act(state)
            
            # Take action in environment and observe result
            next_state, reward, done, _ = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Replay and learn from a batch of experiences
            loss = agent.replay(batch_size)
            if loss is not None:
                episode_losses.append(loss.item())
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Decay epsilon
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
        
        # Save the best model based on reward
        if total_reward > best_reward:
            best_reward = total_reward
            best_model = agent.model.state_dict()
        
        # Update target model every few episodes
        if e % target_update == 0:
            agent.update_target_model()

        total_rewards.append(total_reward)
        if episode_losses:
            avg_loss = np.mean(episode_losses)
            losses.append(avg_loss)
            print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f}, Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.4f}")
        else:
            print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward:.2f}, Loss: N/A, Epsilon: {agent.epsilon:.4f}")

        # Save model every 50 episodes
        if (e + 1) % 50 == 0:
            agent.save(f"dqn_model_{e+1}.pth")
    
    # Save the best model
    torch.save(best_model, "best_dqn_model.pth")
    
    return total_rewards, losses

def plot_metrics(total_rewards, losses):
    episodes = range(1, len(total_rewards) + 1)
    
    # Plot Rewards
    plt.subplot(2, 1, 1)
    plt.plot(episodes, total_rewards, label="Total Rewards")
    plt.title('Rewards Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot Losses
    plt.subplot(2, 1, 2)
    plt.plot(episodes, losses, label="Loss", color='orange')
    plt.title('Loss Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load stock data
    df = pd.read_csv("../../data/NVDA_data.csv")
    
    # Initialize environment and agent
    env = StockTradingEnv(df)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(input_dim=state_size, action_space=action_size, lr=learning_rate, gamma=discount_factor, epsilon=epsilon_start, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay)
    
    
    total_rewards, losses = train_dqn(df, agent, env, episodes=episodes, batch_size=batch_size, target_update=target_update)
    
    
    plot_metrics(total_rewards, losses)
