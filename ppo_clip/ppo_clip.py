import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Define Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
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
        return action_log_probs, torch.squeeze(state_value), entropy

# Hyperparameters
gamma = 0.99            # Discount factor
clip_eps = 0.2          # Clipping parameter for PPO
lr = 3e-4               # Learning rate
epochs = 10             # Number of epochs per update
batch_size = 64         # Batch size for training
update_timestep = 2000  # Number of steps before an update
hidden_dim = 64         # Size of the hidden layer

# Create environment (replace with your environment)
import gym
env = gym.make("CartPole-v1")
input_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Create the Actor-Critic network and optimizer
model = ActorCritic(input_dim, action_dim, hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

def compute_returns(rewards, values, gamma):
    returns = []
    G = 0
    for reward, value in zip(reversed(rewards), reversed(values)):
        G = reward + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    return (returns - returns.mean()) / (returns.std() + 1e-5)

# Training loop
def ppo_train():
    timestep = 0
    while True:
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        state = env.reset()
        for _ in range(update_timestep):
            state = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = model.act(state)
            value = model.critic(state)

            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob)
            states.append(state)
            actions.append(torch.tensor(action))
            dones.append(done)
            state = next_state
            timestep += 1
            
            if done:
                state = env.reset()

        # Compute returns and prepare tensors
        returns = compute_returns(rewards, values, gamma)
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        values = torch.tensor(values)

        # Update the policy for a number of epochs
        for _ in range(epochs):
            new_log_probs, values_pred, entropy = model.evaluate(states, actions)
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute PPO clipped loss
            advantages = returns - values
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Compute critic loss
            critic_loss = nn.MSELoss()(values_pred, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Timestep: {timestep}, Loss: {loss.item()}")

# Run the training
ppo_train()