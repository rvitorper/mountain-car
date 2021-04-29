import copy
import math
import os
import signal
import sys

import gym
import torch
from torch import nn
import numpy as np


env_name = 'MountainCar-v0'
env = gym.make(env_name)
env._max_episode_steps = 500
render = False
observation = env.reset()
last_state = copy.deepcopy(observation)
state_size = len(observation)
best_score = -1e6
best_model = nn.Sequential()

device = "cuda" if torch.cuda.is_available() else "cpu"

n = state_size + 2
m = env.action_space.n
num_hidden_neurons = 50
num_output_neurons = m


def initialize_weights_kaiming(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        torch.nn.init.zeros_(m.bias)


def initialize_weights_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Reinforcement(nn.Module):
    def __init__(self):
        super(Reinforcement, self).__init__()
        self.i = nn.Linear(n, num_hidden_neurons)
        self.l1 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.l2 = nn.Linear(num_hidden_neurons, num_hidden_neurons)
        self.o = nn.Linear(num_hidden_neurons, m)
        initialize_weights_xavier(self.i)
        initialize_weights_kaiming(self.l1)
        initialize_weights_kaiming(self.l2)
        initialize_weights_xavier(self.o)

    def forward(self, x):
        x = self.i(x)
        x = torch.relu(x)
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.o(x)
        return x


model = Reinforcement()
if os.path.exists("state.torch"):
    model.load_state_dict(torch.load("state.torch"))
    print("loading model")
model.to(device)


def get_policy(obs):
    logits = model(obs)
    return torch.distributions.Categorical(logits=logits)


def get_action(obs):
    policy = get_policy(torch.from_numpy(obs).float().unsqueeze(0).to(device))
    value = policy.sample()
    return value.item()


def compute_returns(rewards):
    returns = []
    acc = 0
    for value in rewards[::-1]:
        acc = value + acc
        returns.insert(0, acc)
    return returns


def compute_loss_gradient(obs, action, weights):
    logp = get_policy(obs).log_prob(action)
    return -(logp * weights).mean()


def signal_handler(sig, frame):
    print('saving state')
    print('best loss: {}'.format(best_score))
    torch.save(model.state_dict(), 'models/state-{}-{}-{}-{}.torch'.format(n, m, num_hidden_neurons, env_name))
    sys.exit(0)


def height(observation):
    return math.sin(3 * observation[0]) * 0.45 + 0.55


def energy(observation):
    return observation[1] * observation[1] / 2.0 + gravity * height(observation)


signal.signal(signal.SIGINT, signal_handler)

optim = torch.optim.Adam(model.parameters(), lr=1e-3)

if __name__ == "__main__":
    for epoch in range(200000):
        batch_returns = []
        batch_rewards = []
        episode_count = 3
        observation = env.reset()
        rewards = []
        actions = []
        observations = []
        finished_rendering = False
        gravity = 0.0025
        goal_position = 0.5
        energy_baseline = gravity * height([goal_position, 0])
        for i in range(1, 1000000):
            if render or (epoch % 100 == 0 and not finished_rendering):
                env.render()
            significant_state = np.concatenate((observation, [height(observation), energy(observation)]))
            last_state = observation
            observations.append(significant_state)
            action = get_action(significant_state)
            observation, reward, done, info = env.step(action)
            actions.append(action)
            reward = 0 if reward <= 0 else reward
            reward -= 0.1 * (observation[0] - goal_position) ** 2
            reward += 0.1 * (last_state[0] - goal_position) ** 2
            reward -= 1000.0 * (energy_baseline - energy(observation))
            reward += 1000.0 * (energy_baseline - energy(last_state))
            rewards.append(reward)
            if done:
                episode_reward, episode_len = compute_returns(rewards), len(rewards)
                batch_returns.extend(episode_reward)
                batch_rewards.append(sum(rewards))
                finished_rendering = True
                if len(batch_rewards) >= episode_count:
                    break
                else:
                    observation, done, rewards = env.reset(), False, []

        optim.zero_grad()
        policy_loss = compute_loss_gradient(
            obs=torch.as_tensor(observations, dtype=torch.float32).to(device),
            action=torch.as_tensor(actions, dtype=torch.int32).to(device),
            weights=torch.as_tensor(batch_returns, dtype=torch.float32).to(device),
        )
        policy_loss.backward()
        optim.step()
        if sum(batch_rewards)/len(batch_rewards) > best_score:
            best_score = sum(batch_rewards) / len(batch_rewards)
            best_model = copy.deepcopy(model)
        if epoch % 5 == 0:
            print(policy_loss)
            print('updating epoch {}'.format(epoch))
            print('episodes in batch: {}'.format(len(batch_rewards)))
            print('batch weights: {}'.format(len(batch_returns)))
            print('reward: {}'.format(float(sum(batch_rewards))/len(batch_rewards)))
            print('current loss: {}'.format(policy_loss))
