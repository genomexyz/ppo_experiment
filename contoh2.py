import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Define the Actor-Critic network
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, output_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)

def ppo_clip(env_name, num_episodes, num_steps, epsilon, value_coef, entropy_coef, max_grad_norm, gamma, lmbda, device):
    env = gym.make(env_name)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    model = ActorCritic(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters())

    for episode in range(num_episodes):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        values = []
        log_probs = []

        state = env.reset()

        for step in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_probs, value = model(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            next_state, reward, done, _ = env.step(action.item())

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state

            if len(states) == num_steps:
                next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                _, next_value = model(next_state)
                returns = compute_returns(next_value, rewards, dones, gamma)

                states_tensor = torch.cat(states)
                actions_tensor = torch.cat(actions)
                log_probs_tensor = torch.cat(log_probs)
                returns_tensor = torch.cat(returns).detach()
                values_tensor = torch.cat(values)

                advantages = returns_tensor - values_tensor.detach()

                # Compute surrogate loss
                ratio = torch.exp(log_probs_tensor - log_probs_tensor.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                critic_loss = nn.MSELoss()(returns_tensor, values_tensor)

                # Compute entropy loss
                entropy_loss = -torch.mean(action_probs * torch.log(action_probs + 1e-8))

                # Total loss
                loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                states.clear()
                actions.clear()
                rewards.clear()
                next_states.clear()
                dones.clear()
                values.clear()
                log_probs.clear()

    env.close()

def compute_returns(next_value, rewards, dones, gamma):
    returns = []
    R = next_value
    for reward, done in zip(reversed(rewards), reversed(dones)):
        mask = 1.0 - done
        R = reward + gamma * R * mask
        returns.insert(0, R)
    return returns

# Set hyperparameters
env_name = "CartPole-v1"
num_episodes = 1000
num_steps = 200
epsilon = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
gamma = 0.99
lmbda = 0.95
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run PPO-Clip
ppo_clip(env_name, num_episodes, num_steps, epsilon, value_coef, entropy_coef, max_grad_norm, gamma, lmbda, device)
