import gymnasium as gym
import torch
import torch.nn as nn

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

env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
count_step = 0
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)
   #print('cek rewarad', reward)
   count_step += 1

   if terminated or truncated:
      observation, info = env.reset()
      print('total step', count_step)
      count_step = 0
env.close()