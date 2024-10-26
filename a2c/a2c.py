import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # Shared layer
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)
        state_value = self.critic(x)
        return action_probs, state_value

    def act(self, state):
        action_probs, _ = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs, state_value = self.forward(state)
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        return action_log_probs, state_value, entropy

# Hyperparameters
gamma = 0.99          # Discount factor
lr = 3e-4             # Learning rate
hidden_dim = 128      # Size of hidden layer
num_steps = 5         # Number of steps to accumulate before updating
max_episodes = 1000   # Number of episodes for training

# Initialize environment and model
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize the model and optimizer
model = ActorCritic(input_dim, action_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Compute discounted returns
def compute_returns(rewards, dones, gamma):
    returns = []
    G = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        if done:
            G = 0  # Reset return if episode is done
        G = reward + gamma * G
        returns.insert(0, G)
    return returns

# A2C Training loop
def a2c_train():
    episode_rewards = []
    
    for episode in range(max_episodes):
        state, _ = env.reset()
        log_probs, values, rewards, dones = [], [], [], []
        episode_reward = 0
        
        for _ in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = model.act(state)
            _, state_value = model.forward(state)

            next_state, reward, done, _, _ = env.step(action)
            log_probs.append(log_prob)
            values.append(state_value)
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            state = next_state
            
            if done:
                state, _ = env.reset()
                break

        # Calculate returns
        returns = compute_returns(rewards, dones, gamma)
        returns = torch.tensor(returns)
        values = torch.cat(values).squeeze()

        # Convert lists to tensors
        log_probs = torch.stack(log_probs)
        advantage = returns - values

        # Calculate the loss
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        episode_rewards.append(episode_reward)
        print(f"Episode: {episode}, Reward: {episode_reward}, Loss: {loss.item()}")

        if done:
            state, _ = env.reset()

    env.close()

# Run the training
a2c_train()