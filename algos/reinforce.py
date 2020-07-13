import glob


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


lr = 0.0001

class Policy(nn.Module):

    def __init__(self, state_size, action_size, max_batch_size, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Policy, self).__init__()

        self.state_size = 412
        self.action_size = action_size

        self.fc1 = nn.Linear(self.state_size, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()

        x = x.reshape(-1, self.state_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        return Categorical(action), value


def just_run(policy_nn):
  for epi in range(1000000):
    obs = env.reset()
    for step_numb in range(1000000):
        obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device)
        dist, val = policy_nn(obs_t)

        action = dist.sample()

        obs, reward, done, info = env.step(action.cpu().numpy())

        if done:
            break

        env.render()
        #cv2.imshow('input', obs)
        #cv2.waitKey(1)




def run_reinforce(env, obs_state, num_actions, device, hypeparameters, render=False):


  policy_nn = Policy(412, 9, 1).to(device)

  optimizer = optim.Adam(policy_nn.parameters())

  steps = 0
  action = None

  #import pudb;pudb.set_trace()

  max_episodes = hypeparameters['max_epochs']
  max_steps = hypeparameters['max_steps']

  gamma = hypeparameters['gamma']
  #mem_buff = BasicBuffer(max_steps+1)

  #all_episode_rewards = np.empty((0, 1), int)
  all_episode_rewards = np.asarray([])

  total_steps = 0

  for epi in range(max_episodes):
      obs = env.reset()
      rewards = []
      log_probs = []
      values = []
      dones = []

      for step_numb in range(max_steps):
          total_steps += 1
          obs_t = torch.Tensor(obs).to(device).float()
          #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device)
          dist, val = policy_nn(obs_t)

          action = dist.sample()

          obs, reward, done, info = env.step(action.cpu().item())

          #mem_buff.push(obs, action, rew, next_state, done)

          log_probs.append(dist.log_prob(action).unsqueeze(0))
          values.append(val)
          rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
          dones.append(done)

          if done:
              print('Done epoch {}, max_epochs: {}'.format(epi, hypeparameters['max_epochs']))

              break


      #import pudb;pudb.set_trace()

      final_state = torch.Tensor(obs).to(device).float()
      final_val = policy_nn(obs_t)[1]


      """Computing returns"""
      multiplication_mask = torch.tensor([1 - done for done in dones]).to(device)
      
      R = final_val
      returns = []
      for step in reversed(range(len(rewards))):
          R = rewards[step] + (gamma**(step + 1)) * R * multiplication_mask[step]
          returns.insert(0, R)
          #returns.append(R)

      #import pudb;pudb.set_trace()
     
      log_probs = torch.cat(log_probs)
      returns = torch.cat(returns).detach()
      values = torch.cat(values)

      advantage = returns - values
      
      actor_loss = -(log_probs * advantage.detach()).mean()
      critic_loss = advantage.pow(2).mean()

      optimizer.zero_grad()
      final_loss = actor_loss + critic_loss
      final_loss.backward()
      optimizer.step()
      print('Trained Neural Networks')


  return policy_nn, max_episodes, None