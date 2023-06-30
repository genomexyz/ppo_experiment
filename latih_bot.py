import gymnasium as gym
import torch
import math
import random
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import namedtuple, deque

# Training loop
max_episodes = 1000
max_steps = 1000
gamma = 0.99
epsilon = 0.2
num_steps = 50

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.sigmoid(self.actor(x)), self.critic(x)

def compute_returns(next_value, rewards, dones, gamma):
    returns = []
    #R = next_value
    R = 0
    for reward, done in zip(reversed(rewards), reversed(dones)):
        mask = 1.0 - done
        R = reward + gamma * R * mask
        returns.insert(0, R)
    return returns


env = gym.make("LunarLander-v2", render_mode="human")

# Initialize the policy and optimizer
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
print('cek shape', input_dim, output_dim)

#policy = Policy(input_dim, output_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ActorCritic(input_dim, output_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for iter_ep in range(max_episodes):
    observation, info = env.reset(seed=102)
    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    values = []
    log_probs = []
    for _ in range(max_steps):
        state_tensor = torch.tensor(observation, dtype=torch.float32)
        action_probs, value = model(state_tensor)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        #action = env.action_space.sample()  # this is where you would insert your policy
        observation_next, reward, terminated, truncated, info = env.step(action.item())
        
        done = 0
        if terminated or truncated:
            done = 1
        
        states.append(state_tensor)
        actions.append(action)
        rewards.append(reward)
        next_states.append(observation_next)
        dones.append(done)
        values.append(value)
        log_probs.append(log_prob)

        if len(states) == num_steps:
            log_probs_old = torch.stack(log_probs)
            for _ in range(5):  # PPO update iterations
                state_tensor_ppo = torch.stack(states)
                new_action_prob, new_value = model(state_tensor_ppo)
                
                new_action_dist = Categorical(logits=new_action_prob)
                new_action = new_action_dist.sample()
                new_log_probs = new_action_dist.log_prob(new_action)
                new_log_probs = new_log_probs.squeeze()

                advantages = compute_returns(new_value, rewards, dones, gamma)
                print('cek advan', advantages)
                advantages = torch.tensor(advantages, dtype=torch.float32)
                #print('cek return', returns)
                #print('cek new value', new_values)

                # Calculate the ratio between new and old probabilities
                #print('cek dim log', new_log_probs.size(), log_probs_old.size())
                ratio = torch.exp(new_log_probs - log_probs_old)
                print('cek ratio', ratio)
                # Calculate surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
                surrogate = -torch.min(surr1, surr2)
                # Compute value loss
                print('cek dim adv new_val', advantages.size(), new_value.squeeze().size())
                critic_loss = nn.MSELoss()(advantages, new_value.squeeze())
                print('cek surrogate', critic_loss)
                exit()

        
        #ending move
        # Move to the next state
        observation = observation_next

        if done == 1:
            break

env.close()