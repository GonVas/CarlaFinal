from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from model import RLNet, RLNetCritic, Inceptionv4, RLNetCriticPlaceHolder, RLNetPlaceHolder, RLNetCriticMSG, RLNetMSG
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


#model_pol = RLNet()
#model_crit = RLNet(to_critic=True)



#rand_input = torch.rand(1, 3, 450, 150)

#pol_out = model_pol.forward(rand_input, torch.rand(1, 12), torch.rand(1, 256))
#val = model_crit.forward(rand_input, torch.rand(1, 12), torch.rand(1, 256))

#import pudb; pudb.set_trace()

#model_ft = models.inception_v3(pretrained=True)

#print(model_ft)
#set_parameter_requires_grad(model_ft, feature_extract)
# Handle the auxilary net
#num_ftrs = model_ft.AuxLogits.fc.in_features
#model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
# Handle the primary net
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs,num_classes)
#input_size = 299

 



import glob
import time
import os
from collections import namedtuple, deque
from random import randrange, uniform

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np

import random

import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import multivariate_normal

import scipy.interpolate as interp

from torch.distributions import Categorical, Normal
import numpy as np

from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot

import cv2
from PIL import Image, ImageDraw

import wandb

import random

import torch.multiprocessing as mp
from torch.multiprocessing import Lock

os.environ['OMP_NUM_THREADS'] = '1'


from torch.utils.data.dataset import Dataset


import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse

import CarlaGymEnv

#from pytorch_memlab import MemReporter

class DiskBuffer:

  def __init__(self, max_size, filedir='diskbuffer/'):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)
      self.filedir = filedir
      self.dir_name = filedir + str(int(round(time.time() * 1000)))
      os.mkdir(self.dir_name)
      self.seq = 0
      print('Created folder {}, for disk buffer use, max exps: {}'.format(self.dir_name, max_size))

  def push(self, state, hidden, action, reward, next_hidden, next_state, done, msg_before, msg_next):
      #experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done)
      #import pudb; pudb.set_trace()

      image_np = np.asarray(state[0]*255).astype(np.uint8)
      image_aug_np = np.asarray(state[1])

      next_image_np = np.asarray(next_state[0]*255).astype(np.uint8)
      next_image_aug_np = np.asarray(next_state[1])

      with open(self.dir_name + '/exp_{}.npz'.format(self.seq), 'wb') as f:
        np.savez(f, image_np, image_aug_np, hidden.to("cpu").detach().numpy(), action.to("cpu").detach().numpy(), np.array([reward]), next_hidden.to("cpu").detach().numpy(), next_image_np, next_image_aug_np, done, msg_before.detach().numpy(), msg_next.detach().numpy())
        self.buffer.append(self.dir_name + '/exp_{}.npz'.format(self.seq))
        self.seq += 1


  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []
      hidden_batch = []
      next_hidden_batch = []
      msg_buffer_batchs = []
      msg_before_batch = []
      msg_next_batch = []

      batch = random.sample(self.buffer, batch_size)
      
      for experience_file in batch:
          data = np.load(experience_file, allow_pickle=True)
          image_ts, image_aug_ts, hidden, action, reward, next_hidden, next_image_ts, next_image_aug_ts, done, msg_before, msg_next = data.values()

          state_batch.append((torch.FloatTensor(image_ts)/255, torch.FloatTensor(image_aug_ts)))
          action_batch.append(torch.FloatTensor(action))
          reward_batch.append(reward)
          next_state_batch.append((torch.FloatTensor(next_image_ts)/255, torch.FloatTensor(next_image_aug_ts)))
          done_batch.append(done)
          hidden_batch.append(torch.FloatTensor(hidden))
          next_hidden_batch.append(torch.FloatTensor(next_hidden))

          msg_before_batch.append(msg_before)
          msg_next_batch.append(msg_next)

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch, msg_before_batch, msg_next_batch)


  def __len__(self):
      return len(self.buffer)





class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, hidden, action, reward, next_hidden, next_state, done, msg_before, msg_next):
      experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done, msg_before, msg_next)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []
      hidden_batch = []
      next_hidden_batch = []
      msg_before_batch = []
      msg_next_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, hidden, action, reward, next_hidden, next_state, done, msg_before, msg_next = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)
          hidden_batch.append(hidden)
          next_hidden_batch.append(next_hidden)
          msg_before_batch.append(msg_before)
          msg_next_batch.append(msg_next)


      #import pudb; pudb.set_trace()

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch, msg_before, msg_next)

  def __len__(self):
      return len(self.buffer)






def soft_update(target, source, tau):
    if(target.device != source.device):
        tagert.to(source.device)
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



class SAC():

  def __init__(self, env_action_shape, hyperps, device, train=True):

    self.hyperps = hyperps
    self.env_action_shape = env_action_shape
    self.device = device

    self.actor = RLNetMSG().to(device) #ResNetRLGRU(3, 2, 12)(self.obs_state, self.num_actions).to(device) 

    
    self.critic1 = RLNetMSG().to(device)
    self.critic2 = RLNetMSG().to(device)

    self.targ_critic1 = RLNetMSG().to(device)
    self.targ_critic2 = RLNetMSG().to(device)

    self.scaler = torch.cuda.amp.GradScaler()
    #self.policy_scaler = GradScaler()
    #self.alpha_scaler = GradScaler()


    self.avg_propgation_time = [0, 1]


    print('Before params copy')

    params1 = self.critic1.named_parameters()
    params2 = self.targ_critic1.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(param1.data)


    params1 = self.critic2.named_parameters()
    params2 = self.targ_critic2.named_parameters()

    dict_params2 = dict(params2)

    for name1, param1 in params1:
        if name1 in dict_params2:
          dict_params2[name1].data.copy_(param1.data)

    print('Afeter params copy')

    if(train):
 
        self.critic_optim = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=2e-4)
        self.policy_optim = optim.Adam(self.actor.parameters(), lr=2e-4)

        # entropy temperature
        self.alpha = self.hyperps['alpha']
        self.target_entropy = -torch.prod(torch.Tensor(self.env_action_shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=2e-4)
        
    else:
        self.try_load()




  def critic(self, obs, action, msg):
    c1, msg1 = self.critic1.critic(obs, action, msg) 
    c2, msg2 = self.critic2.critic(obs, action, msg)

    return c1, c2, msg1, msg2


  def critic_target(self, obs, action, msg):
    c1, msg1 = self.targ_critic1.critic(obs, action, msg) 
    c2, msg2 = self.targ_critic2.critic(obs, action, msg)

    return c1, c2, msg1, msg2


  def eval(self):
    self.actor.eval()
    self.critic1.eval()
    self.critic2.eval()


  def train(self):
    self.actor.train()
    self.critic1.train()
    self.critic2.train()


  def sample(self, obs, msg):

    mean, log_std, hidden, msg = self.actor.forward(obs, msg)

    std = log_std.exp()
    normal = Normal(mean, std)
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)

    action = y_t * self.hyperps['action_scale'] #+ self.hyperps['action_bias']
    action[:, 0] += self.hyperps['action_bias']
    log_prob = normal.log_prob(x_t)

    # Enforcing Action Bound
    log_prob -= torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
    log_prob = log_prob.sum(1, keepdim=True)

    mean = torch.tanh(mean) * self.hyperps['action_scale'] + self.hyperps['action_bias']

    entropy = normal.entropy()
    entropy1, entropy2 = entropy[0][0].item(), entropy[0][1].item()

    #print('Std: {:2.3f}, {:2.3f}, log_std: {:2.3f},{:2.3f}, entropy:{:2.3f}, {:2.3f}'.format(std[0][0].item(),std[0][1].item(), log_std[0][0].item(), log_std[0][1].item(), entropy1, entropy2))
    return action, log_prob, mean, std, hidden, msg


  def print_devices(self):
    print('Actor Device: '+str(next(self.actor.parameters()).device))
    print('Critic1 Device: '+str(next(self.critic1.parameters()).device))
    print('Critic2 Device: '+str(next(self.critic2.parameters()).device))
    print('Target Critic1 Device: '+str(next(self.targ_critic1.parameters()).device))
    print('Target Critic2 Device: '+str(next(self.targ_critic2.parameters()).device))

  def mem_report(self):
    self.actor_reporter = MemReporter(self.actor)
    self.critic1_reporter = MemReporter(self.critic1)
    self.critic2_reporter = MemReporter(self.critic2)
    self.targ_critic1_reporter = MemReporter(self.targ_critic1)
    self.targ_critic2_reporter = MemReporter(self.targ_critic2)


  def mem_report_now(self):
    self.actor_reporter.report(verbose=True)
    self.critic1_reporter.report(verbose=True)
    self.critic2_reporter.report(verbose=True)
    self.targ_critic1_reporter.report(verbose=True)
    self.targ_critic2_reporter.report(verbose=True)



  def prob_action(self, obs, action_to_calc):
    # returns probability given policy -> π(action|obs)

    mean, log_std, _, _, _ = self.actor.forward(obs)

    log_std = torch.tanh(log_std)
    log_std = self.hyperps['log_std_min'] + 0.5 * (self.hyperps['log_std_max'] - self.hyperps['log_std_min']) * (log_std + 1)

    std = log_std.exp()
    normal = Normal(mean, std)

    y_t = torch.tanh(action_to_calc)

    action = y_t * self.hyperps['action_scale'] + self.hyperps['action_bias']
    
    log_prob = normal.log_prob(action_to_calc)
    
    # Enforcing Action Bound

    log_prob = log_prob - torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
    log_prob = log_prob.sum(1, keepdim=True)


    return torch.pow(2, log_prob)


  def try_load(self, save_dir='./project/savedmodels/rl/sac/'):
    print('Loading Model')
    paths = glob.glob(save_dir + '*_final.tar') ; step = 0
    if len(paths) > 4:

        self.targ_critic2.load_state_dict(torch.load(paths[-5]))
        self.targ_critic1.load_state_dict(torch.load(paths[-3]))
        self.critic2.load_state_dict(torch.load(paths[-4]))
        self.critic1.load_state_dict(torch.load(paths[-2]))
        self.actor.load_state_dict(torch.load(paths[-1]))



    def save_models(self, episode_numb, save_dir='./savedmodels/rl/sac/'):
            
        torch.save(self.actor.state_dict(), save_dir+'pol_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.critic1.state_dict(), save_dir+'q1_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.critic2.state_dict(), save_dir+'q2_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.targ_critic1.state_dict(), save_dir+'tq1_model.{:.0f}.tar'.format(episode_numb))
        torch.save(self.targ_critic2.state_dict(), save_dir+'tq2_model.{:.0f}.tar'.format(episode_numb))
        print('\n\t{:.0f} Epsiodes: saved model\n'.format(episode_numb))


    def save_models_final(self, save_dir='./project/savedmodels/rl/sac/'):
        
        torch.save(self.actor.state_dict(), save_dir+'pol_model_final.tar')
        torch.save(self.critic1.state_dict(), save_dir+'q1_model_final.tar')
        torch.save(self.critic2.state_dict(), save_dir+'q2_model_final.tar')
        torch.save(self.targ_critic1.state_dict(), save_dir+'tq1_model_final.tar')
        torch.save(self.targ_critic2.state_dict(), save_dir+'tq2_model_final.tar')
        print('\nFinal SAC saved model\n')



  def update(self, memory, updates, expert_data=False):
        batch_size=self.hyperps['batch_size']
        gamma=self.hyperps['gamma']
        tau=self.hyperps['tau']


        #with torch.autograd.set_detect_anomaly(True):

        mem_state_batch, mem_first_hidden_batch, mem_action_batch, mem_reward_batch, mem_next_hidden_batch, mem_next_state_batch, mem_mask_batch, mem_msg_before_batch, mem_msg_after_batch = memory.sample(batch_size)
        mem_states_obs_batch, mem_states_aug_batch = zip(*mem_state_batch)
        mem_next_states_obs_batch, mem_next_states_aug_batch = zip(*mem_next_state_batch)

        first_hidden_batch = torch.stack(mem_first_hidden_batch).to(self.device).squeeze(1)

        state_obs_batch = torch.stack(mem_states_obs_batch).to(self.device).squeeze(1)
        state_aug_batch = torch.stack(mem_states_aug_batch).to(self.device).squeeze(1)

        next_mem_batch = torch.stack(mem_next_hidden_batch).to(self.device).squeeze(1)
        next_next_state_obs_batch = torch.stack(mem_next_states_obs_batch).to(self.device).squeeze(1)
        next_next_state_aug_batch = torch.stack(mem_next_states_aug_batch).to(self.device).squeeze(1)
        #action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
        #action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
        action_batch = torch.stack(mem_action_batch).to(self.device)
        reward_batch = torch.FloatTensor(np.stack(mem_reward_batch)).to(self.device)
        mask_batch = torch.FloatTensor(np.stack(mem_mask_batch)).to(self.device).unsqueeze(1)


        msg_before_batch = torch.FloatTensor(np.stack(mem_msg_before_batch)).to(self.device).squeeze(1)
        msg_after_batch = torch.FloatTensor(np.stack(mem_msg_after_batch)).to(self.device).squeeze(1)
        #import pudb; pudb.set_trace()



        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _, next_hidden, _ = self.sample((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch), msg_before_batch)
            qf1_next_target, qf2_next_target, _, _ = self.critic_target((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch), next_state_action, msg_after_batch)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)
        
            is_cs = 1

            if(expert_data):
                prob_beh = self.prob_action((state_obs_batch, state_aug_batch, first_hidden_batch.detach()), action_batch) + self.hyperps['epsilon']
                prob_expert = torch.ones_like(prob_beh) * 0.90 # Assume expert is almost confident on what it did
                is_cs = torch.min(torch.ones_like(prob_expert)*1.5, prob_expert/prob_beh)

        #import pudb; pudb.set_trace()
        qf1, qf2, _, _ = self.critic((state_obs_batch.detach(), state_aug_batch.detach(), first_hidden_batch.detach()), action_batch.squeeze(1).detach(), msg_before_batch.detach())  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value * is_cs)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value * is_cs)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss 


        """
        scaler.scale(loss).backward()

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()
        """

        start_propag = int(round(time.time() * 1000))
        total_propag_time = 0

        self.critic_optim.zero_grad()

        #self.scaler.scale(qf_loss).backward(retain_graph=True)
        #self.scaler.step(self.critic_optim)

        qf_loss.backward()
        self.critic_optim.step()

        total_propag_time += int(round(time.time() * 1000)) - start_propag 


        pi, log_pi, _, _, _, _ = self.sample((state_obs_batch, state_aug_batch, first_hidden_batch), msg_before_batch)

        qf1_pi, qf2_pi, _, _ = self.critic((state_obs_batch, state_aug_batch, first_hidden_batch), pi, msg_before_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)


        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]


        start_propag = int(round(time.time() * 1000)) 

        self.policy_optim.zero_grad()


        #self.scaler.scale(policy_loss).backward(retain_graph=True)
        #self.scaler.step(self.policy_optim)
        
        policy_loss.backward()
        self.policy_optim.step()

        total_propag_time += int(round(time.time() * 1000)) - start_propag 

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()


        start_propag = int(round(time.time() * 1000)) 

        self.alpha_optim.zero_grad()
        
        #self.scaler.scale(alpha_loss).backward()
        #self.scaler.step(self.alpha_optim)


        alpha_loss.backward()
        self.alpha_optim.step()

        total_propag_time += int(round(time.time() * 1000)) - start_propag 

        self.avg_propgation_time[0] = self.avg_propgation_time[0] + (total_propag_time - self.avg_propgation_time[0]) / self.avg_propgation_time[1]
        self.avg_propgation_time[1] += 1

        self.alpha = self.log_alpha.exp()
        alpha_tlogs = self.alpha.clone() # For TensorboardX logs


        #if updates % self.hyperps['critic_target_update'] == 0:
        #    soft_update(self.targ_critic1, self.critic1, tau)
        #    soft_update(self.targ_critic2, self.critic2, tau)


        #self.scaler.update()
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()






from torchvision import transforms

tfms = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])


def run_sac(env, hyperps, device, rank, shared_msg_list, save_dir='./nvme/', load_buffer_dir='./nvme/diskbuffer/'):

    
    print('Running SAC Simple, no bl no multiagent')

    os.environ['WANDB_MODE'] = 'run'
    wandb.init(config=hyperps, project="gpulab")


    size_image_to_work = (50, 150) 

    mem_max_size = hyperps['maxmem']
    mem_start_thr = 0.01
    mem_train_thr = 0.0125

    memory = DiskBuffer(mem_max_size, filedir=load_buffer_dir)

    print('Batch size: {}'.format(hyperps['batch_size']))

    sac_agent = SAC(env.action_space.shape, hyperps, device)


    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

    #sac_agent.mem_report()

    #sac_agent.mem_report_now()

    #actor_mem_file = open("Actormem.txt", "w")

    #actor_mem_file.write(str(sac_agent.actor_reporter.report(verbose=True)))
    #actor_mem_file.write('Update: 0')

    #time.sleep(15)

    wall_start = time.time()

    total_steps = 0
    updates = 0

    all_actions = []
    all_pol_stats = []
    all_stds = []
    all_means = []

    all_rewards = []
    all_scenario_wins_rewards = []
    all_final_rewards = []
    all_q_vals = []

    to_plot = []

    #w_vel, w_t, w_dis, w_col, w_lan, w_waypoint = 0.010, 1, 10, 1, 1, 10

    w_vel, w_t, w_dis, w_col, w_lan, w_waypoint = 4.5, 40, 5, 10, 10, 50
    #ok 1, 10, 5, 10, 10, 10
    rewards_weights = [w_vel, w_t, w_dis, w_col, w_lan, w_waypoint]

    wandb.run.summary["reward_weights"] = rewards_weights


    old_hidden = torch.zeros(1, 256).to(device)

    done = False

    cumulative_reward = 0

    for epi in range(hyperps['max_epochs']):

        obs = env.reset() # 300, 900, 3 -> 1, 3, 300, 900

        old_obs = (F.interpolate(tfms(torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device).squeeze(0)).unsqueeze(0), size=
                  size_image_to_work), torch.FloatTensor(obs[1]).to(device))

        total_steps += 1

        print('Epoch: {}, Max Epochs: {}, max steps: {}, total_steps: {}, cumulative_reward: {}'.format(epi, hyperps['max_epochs'], hyperps['max_steps'], total_steps, cumulative_reward))
        
        cumulative_reward = 0

        for step_numb in range(hyperps['max_steps']):

            action, log_prob, mean, std, hidden = None, None, None, None, None
            sac_agent.eval()

            #sac_agent.print_devices()
            
            action, log_prob, mean, std, hidden, msg = sac_agent.sample((old_obs[0], old_obs[1], old_hidden), shared_msg_list[(rank + 1) % 2].to(device))

            shared_msg_list[rank] = msg

            yield msg 

            #import pudb; pudb.set_trace()
            q1, q2, _, _ = sac_agent.critic((old_obs[0], old_obs[1], old_hidden), action, shared_msg_list[(rank + 1) % 2].to(device))

            obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])

            reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6

            cumulative_reward += reward

            if(total_steps % 100 == 0):
              print('Final Sum Reward: {:.5f}'.format(reward))
              print('Total steps {}, max_steps: {}'.format(total_steps, hyperps['max_steps']))


            wandb.log({"final_r": reward, 'action':action.cpu(), 'log_prob':log_prob.cpu(), 'mean':mean.cpu(), 'std':std.cpu()})

            if(info != None):
                print(info)
           

            obs = (F.interpolate(tfms(torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device).squeeze(0)).unsqueeze(0), size=
                    size_image_to_work), torch.FloatTensor(obs[1]).to(device))

            memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), old_hidden.to("cpu"), action.to("cpu"), reward, hidden.to("cpu"), (obs[0].to("cpu"), obs[1].to("cpu")), done, shared_msg_list[(rank + 1) % 2].to("cpu"), msg.to("cpu"))



            old_obs = obs
            old_hidden = hidden

            total_steps += 1
            

            if done:
                print('In SAC Done: {}, step : {}, epoch: {}'.format(len(memory), step_numb, epi))

                if(info['scen_sucess'] != None and info['scen_sucess'] == 1):
                    all_scenario_wins_rewards.append(1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})

                elif (info['scen_sucess'] != None and info['scen_sucess'] == -1):
                    all_scenario_wins_rewards.append(-1)
                    wandb.log({"all_scenario_wins_rewards": all_scenario_wins_rewards})

                break


            to_train_mem = memory
            expert_data = False
              

            if len(to_train_mem) > hyperps['batch_size']*5 and len(memory) > mem_train_thr*mem_max_size:
                
                sac_agent.train()
                #import pudb; pudb.set_trace()
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(to_train_mem, updates, expert_data)

                print('Trained Nets')

                #sac_agent.actor_reporter.report()
                #input()
                #actor_mem_file.write('Update: {}'.format(updates))

                if(total_steps != 0 and total_steps % 10 == 0):
                    print('Trained, len of mem: {}'.format(len(memory)))
                    print('Avg: propagation training time: ' + str(sac_agent.avg_propgation_time))
                wandb.log({"critic_1_loss": critic_1_loss, "critic_2_loss": critic_2_loss, "policy_loss": policy_loss, 'ent_loss':ent_loss, 'alpha':alpha})

                updates += 1

            if(updates != 0 and updates % 5_000 == 0):
                print('Saving')
                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.actor.state_dict(),
                        }, save_dir+'sac_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic1.state_dict(),
                        }, save_dir+'sac_c1_model_{}.tar'.format(total_steps))

                torch.save({
                        'steps': total_steps,
                        'model_state_dict': sac_agent.critic2.state_dict(),
                        }, save_dir+'sac_c2_model_{}.tar'.format(total_steps))

                wandb.save(save_dir+'sac_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c1_model_{}.tar'.format(total_steps))
                wandb.save(save_dir+'sac_c2_model_{}.tar'.format(total_steps))


            if(total_steps >= hyperps['max_steps']):
              break
        
        wandb.log({'Episode_Cumulative_Reward' : cumulative_reward, 'epi':epi})


        if(total_steps >= hyperps['max_steps']):
          break
    actor_mem_file.close()
    print('Final Save')
    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.actor.state_dict(),
            }, save_dir+'final_sac_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic1.state_dict(),
            }, save_dir+'final_sac_c1_model.tar')

    torch.save({
            'steps': total_steps,
            'model_state_dict': sac_agent.critic2.state_dict(),
            }, save_dir+'final_sac_c2_model.tar')

    wandb.save(save_dir+'final_sac_model.tar')
    wandb.save(save_dir+'final_sac_c1_model.tar')
    wandb.save(save_dir+'final_sac_c2_model.tar')

    wandb.join()

    return sac_agent



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == "__main__":

    

    parser = argparse.ArgumentParser(description='Carla RL and VAE')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--vae', action='store_true', default=False,
                        help='Train and use VAE.')
    parser.add_argument('--epochs', type=int, default=0, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument("--sparse", type=str2bool, nargs='?',
                            const=True, default=False,
                            help="Activate Sparse Rewards.")
    parser.add_argument("--rl", help="Rl algorithm to use",type=str, default='dqn')
    parser.add_argument('--maxram', type=int, default=5, metavar='N',
                        help='Max ram usage for the dataset in GB, its approximate and multiple of batch size, final size is +- 2GB')
    parser.add_argument('--production', action='store_true', default=False,
                        help='Turn this on for a server production run, and default is for local env')

    args = parser.parse_args()



    hyperps = {}

    save_dir = './'
    load_buffer_dir = './diskbuffer/'
    human_samples = './human_samples_lidar/'

    if(args.production):
        #args.batch_size = 4
        #total_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2 
        #args.batch_size = int((total_mem_mb - 3000)/260)
        args.batch_size = 16
        args.epochs = 2_000_000 if args.epochs == 0 else args.epochs 
        args.maxram = 8
        hyperps['maxmem'] = 500_000 # 10k -> 15GB, 500k -> 750GB
        hyperps['max_steps'] = 2_000_000

        os.environ['WANDB_MODE'] = 'run'

        save_dir = './nvme/'
        load_buffer_dir = './nvme/diskbuffer/'
        human_samples = './nvme/human_samples_lidar/'

        if not os.path.exists('./nvme/'):
            os.makedirs('./nvme/')

        if not os.path.exists('./nvme/diskbuffer/'):
            os.makedirs('./nvme/diskbuffer/')
    else:
        args.batch_size = 2
        # 1650MB cuda for batch 2, 1910 for batch 3, 2130 for batch 4, ~280MB per increase in batch size 
        args.epochs = 10000 if args.epochs == 0  else args.epochs 
        args.maxram = 5
        args.no_cuda = False
        hyperps['maxmem'] = 1000
        hyperps['max_steps'] = 17_500
        os.environ['WANDB_MODE'] = 'run'

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    print(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    print('Cuda exists: {}, Cuda version {}'.format(torch.cuda.is_available(), torch.version.cuda))
    print('Running Rl algorithm: {}'.format(args.rl))

    
    hyperps['max_epochs'] = args.epochs
    
    hyperps['log_interval'] = int(args.epochs/10)
    hyperps['maxram'] = args.maxram
    hyperps['q_lr'] = 0.0003
    hyperps['a_lr']= 0.0003
    hyperps['action_scale'] = 2.5
    hyperps['action_bias'] = 1.5
    hyperps['hidden'] = 256
    hyperps['rnn_steps'] = 4
    hyperps['batch_size'] = args.batch_size
    hyperps['bl_batch_size'] = args.batch_size
    hyperps['start_steps'] = args.batch_size*3

    hyperps['updates_per_step'] = 1
    hyperps['update_every'] = 1

    hyperps['alpha'] = 0.1
    hyperps['gamma'] = 0.99
    hyperps['tau'] = 0.005
    hyperps['critic_target_update'] = 4
    hyperps['seq'] = 4
    hyperps['update_step'] = 4
    hyperps['betas'] = (0.9, 0.999)


    hyperps['log_std_max'] = 0.1
    hyperps['log_std_min'] = 0.01
    hyperps['epsilon'] = 1e-6
    

 
    os.environ['WANDB_MODE'] = 'run'
    os.environ['WANDB_API_KEY'] = "4b3486db7da0dff72366b5e2b6b791ae41ae3b9f"


    #env = CarlaGymEnv.CarEnv(0, render=True, step_type="other", benchmark="Simple", auto_reset=False, discrete=False, sparse=args.sparse, dist_reward=True, display2d=False)


    #final_nn = sac_simple_channel.run_sac_dist(hyperps)


    #env = CarlaGymEnv.CarEnvScenario(0)
    #(env, hyperps, shared_model, shared_optim, sample_buffer=None, device=torch.device("cpu"), render=True, metrified=True, save_dir='./', load_buffer_dir='./human_samples/')
    #final_nn = sac_simple_channel.run_sac(env, hyperps, None, None, device=device, save_dir=save_dir, load_buffer_dir=load_buffer_dir)
    

    #final_nn = sac_simple_channel.run_sac_dist(hyperps, human_samples=human_samples, save_dir=save_dir, double_phase=True, load=True, load_buffer_dir=load_buffer_dir)


    #env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True, metrified=True, save_dir='./', load_buffer_dir='./human_samples/'):
    #final_nn = singular_rl.run_sac(env, ((300, 900), 3), 2, hyperps, device=device, save_dir=save_dir, load_buffer_dir=load_buffer_dir)



    env = CarlaGymEnv.CarEnv(0, render=False, step_type="other", benchmark="STDRandom", auto_reset=False, discrete=False, sparse=False, dist_reward=True, display2d=False, distributed=False)
    time.sleep(8)
    env1 = CarlaGymEnv.CarEnv(-1, render=False, step_type="other", benchmark="STDRandom", auto_reset=False, discrete=False, sparse=False, dist_reward=True, display2d=False, distributed=False)


    shared_msg_list = [torch.zeros(1, 32).float(), torch.zeros(1, 32).float()]

    agent_1_gen = run_sac(env, hyperps, torch.device("cuda"), 0, shared_msg_list)
    agent_2_gen = run_sac(env1, hyperps, torch.device("cuda"), -1, shared_msg_list)


    while True:
    
        msg1 = next(agent_1_gen)
        print("MSG1: " + str(msg1))

        msg2 = next(agent_2_gen)
        print('MSG2: ' + str(msg2))


# Definir uma metodoligia:
# - Que scenarios correr
# - Quantas vezes 
# - Metricas para cada scenario
# - Scenario random, metrica ate ao objectivo
#
#


