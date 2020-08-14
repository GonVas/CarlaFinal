import glob
import time
from collections import namedtuple, deque
from random import randrange, uniform


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np



import glob



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

import wandb


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


    def __len__(self):
      return len(self.states)


class Policy(nn.Module):

    def __init__(self, state_size, action_size, aditional_aug, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(Policy, self).__init__()

        self.state_size, channels_in = state_size
        self.action_size = action_size
        self.aditional_aug = aditional_aug

        #self.max_batch_size = max_batch_size

        self.conv1 = nn.Conv2d(channels_in, conv_channels, kernel_size, stride=1)

        self.size_now = self.conv_output_shape(self.state_size) 

        self.pool1 = nn.MaxPool2d(2, 2)

        self.size_now = (int(self.size_now[0]/2), int(self.size_now[1]/2))

        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size)

        self.size_now = self.conv_output_shape(self.size_now)

        self.pool2 = nn.MaxPool2d(2, 2)

        self.size_now = int(self.size_now[0]/2) * int(self.size_now[1]/2) * conv_channels*2

        self.fc1 = nn.Linear(self.size_now + aditional_aug, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        self.final_layer = nn.Linear(size_3, action_size)

        self.critic = nn.Linear(size_3, 1)


    def forward(self, x):
        #import pudb; pudb.set_trace()

        x, aditional = x

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        
        x = x.reshape(-1, self.size_now)

        #import pudb; pudb.set_trace()
        x = torch.cat((x, aditional.reshape(-1, 12)), dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        action = F.softmax(self.final_layer(x))

        value = F.relu(self.critic(x))

        return Categorical(action), value


    def conv_output_shape(self, h_w, kernel_size=3, stride=1, pad=0, dilation=1):
        """
        Utility function for computing output of convolutions
        takes a tuple of (h,w) and returns a tuple of (h,w)
        """
        
        if type(h_w) is not tuple:
            h_w = (h_w, h_w)
        
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        
        if type(stride) is not tuple:
            stride = (stride, stride)
        
        if type(pad) is not tuple:
            pad = (pad, pad)
        
        h = (h_w[0] + (2 * pad[0]) - (dilation * (kernel_size[0] - 1)) - 1)// stride[0] + 1
        w = (h_w[1] + (2 * pad[1]) - (dilation * (kernel_size[1] - 1)) - 1)// stride[1] + 1
        
        return h, w



class PolicySimple(nn.Module):

    def __init__(self, state_size, action_size, max_batch_size, conv_channels=6, kernel_size=3, size_1=32, size_2=64, size_3=32):
        super(PolicySimple, self).__init__()

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


def ppo_update(memory, new_policy, old_policy, MseLoss, optimizer, device, hypeparameters):

  lr = hypeparameters['q_lr']
  betas = hypeparameters['betas']
  gamma = hypeparameters['gamma']
  K_epochs = hypeparameters['K_epochs']
  eps_clip = hypeparameters['eps_clip']

  rewards = []
  discounted_reward = 0
  for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
      if is_terminal:
          discounted_reward = 0
      discounted_reward = reward + (gamma * discounted_reward)
      rewards.insert(0, discounted_reward)
  
  # Normalizing the rewards:
  rewards = torch.tensor(rewards).to(device).float()
  rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
  
  # convert list to tensor

  #old_states = torch.stack(memory.states).to(device).detach()
  old_states = list(zip(*memory.states))
  old_actions = torch.stack(memory.actions).to(device).detach()
  old_logprobs = torch.stack(memory.logprobs).to(device).detach()
  
  losses = []
  entropies = []

  #import pudb; pudb.set_trace()

  for _ in range(K_epochs):

      dist_action_wouldtaken, state_values = new_policy((torch.stack(old_states[0]).squeeze(1), torch.stack(old_states[1])))

      logprobs = dist_action_wouldtaken.log_prob(old_actions)
      dist_entropy = dist_action_wouldtaken.entropy()
      state_values = torch.squeeze(state_values)

      #logprobs, state_values, dist_entropy = new_policy.evaluate(old_states, old_actions)
      
      # Finding the ratio (pi_theta / pi_theta__old):
      ratios = torch.exp(logprobs - old_logprobs.detach())
          
      # Finding Surrogate Loss:
      advantages = rewards - state_values.detach()
      surr1 = ratios * advantages
      surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
      loss = -torch.min(surr1, surr2) + 0.5*MseLoss(state_values, rewards) - 0.01*dist_entropy

      # take gradient step
      optimizer.zero_grad()
      loss.mean().backward()
      losses.append(loss.mean())
      entropies.append(dist_entropy.mean())
      optimizer.step()
  
  # Copy new weights into old policy:
  old_policy.load_state_dict(new_policy.state_dict())
  return losses, entropies


def run(env, max_episodes=100, max_steps=100):

  for epi in range(max_episodes):
    epi_reward = 0
    for step in range(max_steps):

      action = env.action_space.sample()
      obs, reward, done, _ = env.step(action)   

      epi_reward += reward
      if(done):
        print('Done episode {:3d}, with total reward: {:2.2f}'.format(epi, epi_reward))
        break



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
        cv2.imshow('input', obs)
        cv2.waitKey(1)

#746959


def run_pg(env, obs_state, num_actions, device, hypeparameters, render=False):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ppo_net = PolicySimple(412, 9, 1).to(device).float()
    old_ppo_net = PolicySimple(412, 9, 1).to(device).float()
    old_ppo_net.load_state_dict(ppo_net.state_dict())

    mseloss = nn.MSELoss()
    optimizer = optim.Adam(ppo_net.parameters())


    action = None

    memory = Memory()

    all_episode_rewards = np.asarray([])

    total_steps = 0

    render = True

    for epi in range(hypeparameters['max_epochs']):
        
        obs = env.reset()
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float()
        obs_t = torch.Tensor(obs).to(device).float()

        epi_rewards = 0

        for step_numb in range(hypeparameters['max_steps']):

            #import pudb; pudb.set_trace()

            dist, val = ppo_net(obs_t)

            action = dist.sample()

            obs, reward, done, info = env.step(action.cpu().item())

            epi_rewards += reward

            #obs_t = torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float()
            obs_t = torch.Tensor(obs).to(device).float()

            memory.actions.append(action)
            memory.states.append(obs_t)
            memory.logprobs.append(dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            total_steps += 1

            if done:
                print('Done epoch {}, max_epochs: {}, len memory: {}'.format(epi, hypeparameters['max_epochs'], len(memory)))
                break

  
            # update if its time
            if total_steps % hypeparameters['update_step'] == 0:
                losses, entropies = ppo_update(memory, ppo_net, old_ppo_net, mseloss, optimizer, device, hypeparameters)
                memory.clear_memory()
                print('Trained Neural Networks')

                print('Lossed mean: {:2.4f}, entropies mean : {:2.4f}, episode reward: {:3.4f}'.format(sum(losses)/len(losses), sum(entropies)/len(entropies), epi_rewards))
            

            #running_reward += reward
            if render:
                env.render()
                

        #if((epi + 1) % 500 == 0):
          #save_models(policy_nn, epi)


  


    stats = None
    return ppo_net, hypeparameters['max_epochs'], stats



def run_ppo_convs(env, device, hypeparameters, save_dir, render=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #wandb.init(config=hypeparameters)

    ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()

    old_ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()
    old_ppo_net.load_state_dict(ppo_net.state_dict())

    #wandb.watch(ppo_net)

    mseloss = nn.MSELoss()
    optimizer = optim.Adam(ppo_net.parameters())

    action = None

    memory = Memory()

    all_episode_rewards = np.asarray([])

    total_steps = 0

    render = True

    last_loss = 0

    for epi in range(hypeparameters['max_epochs']):
        
        obs = env.reset()
        obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
        #obs_t = torch.Tensor(obs).to(device).float()

        epi_rewards = 0


        for step_numb in range(hypeparameters['max_steps']):

            #import pudb; pudb.set_trace()

            dist, val = ppo_net(obs_t)

            action = dist.sample()

            obs, reward, done, info = env.step(action.cpu().item())

            epi_rewards += reward

            obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
            #obs_t = torch.Tensor(obs).to(device).float()

            memory.actions.append(action)
            memory.states.append(obs_t)
            memory.logprobs.append(dist.log_prob(action))
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            total_steps += 1

            if done:
                print('Done epoch {}, max_epochs: {}, len memory: {}'.format(epi, hypeparameters['max_epochs'], len(memory)))
                #wandb.log({'episode_steps':step_numb})
                break

  
            # update if its time
            if total_steps % hypeparameters['update_step'] == 0:
                losses, entropies = ppo_update(memory, ppo_net, old_ppo_net, mseloss, optimizer, device, hypeparameters)
                memory.clear_memory()
                print('Trained Neural Networks')
                #wandb.log({"mean_loss": sum(losses)/len(losses), "mean_entropy": sum(entropies)/len(entropies)}, {"episode_reward": epi_rewards})

                print('Lossed mean: {:2.4f}, entropies mean : {:2.4f}, episode reward: {:3.4f}'.format(sum(losses)/len(losses), sum(entropies)/len(entropies), epi_rewards))
                last_loss = sum(losses)/len(losses)
            
            #running_reward += reward
            #if render:
            #    env.render()

            if (total_steps != 0 and total_steps % 500 == 0):
              env.record_video()



            #if(total_steps == 100):
            #  import pudb; pudb.set_trace()
                

        #if((epi + 1) % 50 == 0):
        #  save_models(policy_nn, epi)
  

    torch.save({
            'steps': total_steps,
            'model_state_dict': ppo_net.state_dict(),
            'loss': last_loss,
            }, save_dir+'final_model.tar')

    #wandb.save(save_dir+'final_model.tar')

    return ppo_net, last_loss



import cv2

def run_scenario_convs(env, path_to_nn, render=False):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  checkpoint = torch.load(path_to_nn)

  ppo_net = Policy(((300, 900), 3), 9, 12).to(device).float()
  ppo_net.load_state_dict(checkpoint['model_state_dict'])

  policy = ppo_net


  obs = env.reset()
  obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())
  #obs_t = torch.Tensor(obs).to(device).float()



  #writer = cv2.VideoWriter('test1.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25, (900, 300), False)

  # Define the codec and create VideoWriter object
  #fourcc = cv2.VideoWriter_fourcc(*'DIVX')

  #out = cv2.VideoWriter('output.avi', -1, 20.0, (900,300))

  fourcc = cv2.VideoWriter_fourcc(*'MP42')
  video = cv2.VideoWriter('./test.avi', fourcc, float(20), (900, 300))


  while True:

    dist, val = policy(obs_t)

    action = dist.sample()

    action = int(action/3), action%3

    thrt_action, steer_action = action
    #Discrete(3) -> 0, 1, 2 -> transform to -1, 0, 1
    thrt_action -= 1
    steer_action -= 1



    obs, reward, done, info = env.step((torch.FloatTensor([thrt_action]), steer_action.cpu()))


    #render = True
    #if render:
    #  cv2.imshow('rl_agent_sensors', obs[0])
    #  cv2.waitKey(1)

    frame_np = obs[0]*255
    frame_np = frame_np.astype(np.uint8)
    video.write(frame_np)

    obs_t = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).to(device).float(), torch.Tensor(obs[1]).to(device).float())


  # Release everything if job is finished

  video.release()