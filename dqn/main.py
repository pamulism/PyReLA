
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from dqn import *
import dqn
import trainer

def main():
    # Create the environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    episode_term = 1000
    # Create the DQN agent
    agent = dqn.DQNAgent(state_dim, action_dim)

    # Train the agent
    rewards = trainer.train_dqn_agent(env, agent, episode_term)

    num_episodes = list(range(1,episode_term+1))

    num_ticks = 300 / 15
    width = max(8, num_ticks * 0.65)
    plt.figure(figsize=(width,6))
    plt.plot(num_episodes, rewards, color='r')
    plt.xticks(np.arange(0,episode_term+1, num_ticks))
    plt.title('Total Rewards by DQN')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.legend()

    #plt.grid(True)
    plt.savefig('dqn_rewards_300.png')
    plt.show()

if __name__ == "__main__":
    main()