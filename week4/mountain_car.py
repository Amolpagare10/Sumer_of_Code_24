import gym
import numpy as np
import matplotlib.pyplot as plt
import os

class QAgent:
    def __init__(self, env: str) -> None:
        self.env_name = env
        self.env = gym.make(env)
        self.state = self.env.reset()[0]
        
        self.observation_space_size = len(self.state)
        self.actions = self.env.action_space.n
        
        self.observation_space_low = self.env.observation_space.low
        self.observation_space_high = self.env.observation_space.high
        
        self.discrete_sizes = [25, 25]
        self.alpha = 0.1
        self.gamma = 0.95
        
        self.num_train_episodes = 25000
        self.epsilon = 1
        self.num_episodes_decay = 15000
        self.epsilon_decay = self.epsilon / self.num_episodes_decay
        
        self.q_table = np.random.uniform(low=-2, high=0, size=(*self.discrete_sizes, self.actions))
        
        self.rewards = []

    def get_state_index(self, state):
        state_idx = []
        for i in range(self.observation_space_size):
            discrete_idx = int((state[i] - self.observation_space_low[i]) / (self.observation_space_high[i] - self.observation_space_low[i]) * self.discrete_sizes[i])
            discrete_idx = min(self.discrete_sizes[i] - 1, max(0, discrete_idx))
            state_idx.append(discrete_idx)
        return tuple(state_idx)

    def update(self, state, action, reward, next_state, is_terminal):
        state_idx = self.get_state_index(state)
        next_state_idx = self.get_state_index(next_state)
        
        if is_terminal:
            self.q_table[state_idx][action] = reward
        else:
            max_next_q = np.max(self.q_table[next_state_idx])
            current_q = self.q_table[state_idx][action]
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state_idx][action] = new_q

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_idx = self.get_state_index(self.state)
            return np.argmax(self.q_table[state_idx])
    
    def env_step(self):
        action = self.get_action()
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        self.update(self.state, action, reward, next_state, terminated and not truncated)
        
        self.state = next_state
        
        return terminated or truncated, reward
    
    def agent_eval(self):
        eval_env = gym.make(self.env_name, render_mode="human")
        done = False
        eval_state = eval_env.reset()[0]
        while not done:
            state_idx = self.get_state_index(eval_state)
            action = np.argmax(self.q_table[state_idx])
            next_state, reward, terminated, truncated, info = eval_env.step(action)
            
            eval_env.render()
            
            done = terminated or truncated
            eval_state = next_state
            
    def train(self, eval_intervals, save_interval, plot_interval):
        for episode in range(1, self.num_train_episodes + 1):
            done = False
            episode_reward = 0
            while not done:
                done, reward = self.env_step()
                episode_reward += reward
            self.rewards.append(episode_reward)
            self.state = self.env.reset()[0]
            
            self.epsilon = max(0, self.epsilon - self.epsilon_decay)
            
            if episode % eval_intervals == 0:
                self.agent_eval()
                
            if episode % save_interval == 0:
                self.save_q_table(f'q_table_episode_{episode}.npy')
                
            if episode % plot_interval == 0:
                self.plot_rewards(interval=plot_interval)

    def save_q_table(self, filename):
        if not os.path.exists('q_tables'):
            os.makedirs('q_tables')
        filepath = os.path.join('q_tables', filename)
        np.save(filepath, self.q_table)
        print(f"Q-table saved to {filepath}")

    def plot_rewards(self, interval=100):
        plt.plot(self.rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards over Episodes')
        plt.savefig(f'rewards_plot_{len(self.rewards)//interval * interval}.png')
        plt.show()

if __name__ == "__main__":
    agent = QAgent("MountainCar-v0")
    agent.train(eval_intervals=1000, save_interval=5000, plot_interval=1000)
