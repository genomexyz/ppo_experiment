import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# Define a simple policy network
class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = self.fc2(x)
        return x

# Create the environment
env = gym.make('CartPole-v1')

# Initialize the policy and optimizer
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
policy = Policy(input_dim, output_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

# Define the PPO update function
def update_policy(states, actions, rewards, log_probs, values, gamma, epsilon):
    returns = []
    discounted_reward = 0
    for reward in reversed(rewards):
        discounted_reward = reward + gamma * discounted_reward
        returns.insert(0, discounted_reward)

    returns = torch.tensor(returns)
    advantages = returns - values

    for _ in range(5):  # PPO update iterations
        # Calculate new log probabilities and values
        new_log_probs = policy(states)
        dist = Categorical(logits=new_log_probs)
        new_values = dist.probs.squeeze()

        # Calculate the ratio between new and old probabilities
        ratio = torch.exp(new_log_probs - log_probs)

        # Calculate surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate = -torch.min(surr1, surr2)

        # Update policy using the surrogate objective
        optimizer.zero_grad()
        loss = surrogate.mean()
        loss.backward()
        optimizer.step()

# Training loop
max_episodes = 1000
max_steps = 200
gamma = 0.99
epsilon = 0.2

for episode in range(max_episodes):
    state = env.reset()
    log_probs = []
    values = []
    rewards = []
    #print('cek state', state[0])

    for steps in range(max_steps):
        state_tensor = torch.tensor(state[0], dtype=torch.float32)
        action_probs = policy(state_tensor)
        dist = Categorical(logits=action_probs)
        action = dist.sample()

        next_state, reward, done, _ = env.step(action.item())

        log_prob = dist.log_prob(action)
        log_probs.append(log_prob)
        values.append(action_probs.squeeze())
        rewards.append(reward)

        state = next_state

        if done:
            break

    update_policy(torch.stack(log_probs), torch.stack(values), torch.tensor(rewards), gamma, epsilon)

# Use the trained policy
state = env.reset()
done = False

while not done:
   
    state = torch.tensor(state, dtype=torch.float32)
    action_probs = policy(state)
    dist = Categorical(logits=action_probs)
    action = dist.sample()
    next_state, reward, done, _ = env.step(action.item())

    state = next_state

    if done:
        break

env.close()
