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


from architectures_no_msg import ResNetRLGRU, ResNetRLGRUCritic


import CarlaGymEnv



class DiskBuffer:

  def __init__(self, max_size, filedir='diskbuffer/'):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)
      self.filedir = filedir
      self.dir_name = filedir + str(int(round(time.time() * 1000)))
      os.mkdir(self.dir_name)
      self.seq = 0
      print('Created folder {}, for disk buffer use, max exps: {}'.format(self.dir_name, max_size))

  def push(self, state, hidden, action, reward, next_hidden, next_state, done):
      #experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done)
      #import pudb; pudb.set_trace()

      image_np = np.asarray(state[0]*255).astype(np.uint8)
      image_aug_np = np.asarray(state[1])

      next_image_np = np.asarray(next_state[0]*255).astype(np.uint8)
      next_image_aug_np = np.asarray(next_state[1])

      with open(self.dir_name + '/exp_{}.npz'.format(self.seq), 'wb') as f:
        np.savez(f, image_np, image_aug_np, hidden.to("cpu").detach().numpy(), action.to("cpu").detach().numpy(), np.array([reward]), next_hidden.to("cpu").detach().numpy(), next_image_np, next_image_aug_np, done)
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

      batch = random.sample(self.buffer, batch_size)
      
      for experience_file in batch:
          data = np.load(experience_file, allow_pickle=True)
          image_ts, image_aug_ts, hidden, action, reward, next_hidden, next_image_ts, next_image_aug_ts, done = data.values()

          state_batch.append((torch.FloatTensor(image_ts)/255, torch.FloatTensor(image_aug_ts)))
          action_batch.append(torch.FloatTensor(action))
          reward_batch.append(reward)
          next_state_batch.append((torch.FloatTensor(next_image_ts)/255, torch.FloatTensor(next_image_aug_ts)))
          done_batch.append(done)
          hidden_batch.append(torch.FloatTensor(hidden))
          next_hidden_batch.append(torch.FloatTensor(next_hidden))

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch)


  def __len__(self):
      return len(self.buffer)





class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, hidden, action, reward, next_hidden, next_state, done):
      experience = (state, hidden, action, np.array([reward]), next_hidden, next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []
      hidden_batch = []
      next_hidden_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, hidden, action, reward, next_hidden, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)
          hidden_batch.append(hidden)
          next_hidden_batch.append(next_hidden)


      #import pudb; pudb.set_trace()

      return (state_batch, hidden_batch, action_batch, reward_batch, next_hidden_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)






def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)



class SAC():

  def __init__(self, env_action_shape, hyperps, device, train=True):

    self.hyperps = hyperps
    self.env_action_shape = env_action_shape
    self.device = device


    self.actor = ResNetRLGRU(3, 2, 12).to(device) #ResNetRLGRU(3, 2, 12)(self.obs_state, self.num_actions).to(device) 

    self.critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

    self.targ_critic1 = ResNetRLGRUCritic(3, 2, 12).to(device)
    self.targ_critic2 = ResNetRLGRUCritic(3, 2, 12).to(device)

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




  def critic(self, obs, action):
    c1 = self.critic1(obs, action) 
    c2 = self.critic2(obs, action)

    return c1, c2


  def critic_target(self, obs, action):

    c1 = self.targ_critic1(obs, action)
    c2 = self.targ_critic2(obs, action)

    return c1, c2


  def eval(self):
    self.actor.eval()
    self.critic1.eval()
    self.critic2.eval()


  def train(self):
    self.actor.train()
    self.critic1.train()
    self.critic2.train()


  def sample(self, obs):

    mean, log_std, hidden = self.actor.forward(obs)
    
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
    return action, log_prob, mean, std, hidden



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


        with torch.autograd.set_detect_anomaly(True):

            mem_state_batch, mem_first_hidden_batch, mem_action_batch, mem_reward_batch, mem_next_hidden_batch, mem_next_state_batch, mem_mask_batch = memory.sample(batch_size)
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


            with torch.no_grad():
                next_state_action, next_state_log_pi, _, _, next_hidden = self.sample((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch))
                qf1_next_target, qf2_next_target = self.critic_target((next_next_state_obs_batch, next_next_state_aug_batch, next_mem_batch), next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)
            
                is_cs = 1

                if(expert_data):
                    prob_beh = self.prob_action((state_obs_batch, state_aug_batch, first_hidden_batch.detach()), action_batch) + self.hyperps['epsilon']
                    prob_expert = torch.ones_like(prob_beh) * 0.90 # Assume expert is almost confident on what it did
                    is_cs = torch.min(torch.ones_like(prob_expert)*1.5, prob_expert/prob_beh)

            qf1, qf2 = self.critic((state_obs_batch.detach(), state_aug_batch.detach(), first_hidden_batch.detach()), action_batch.detach())  # Two Q-functions to mitigate positive bias in the policy improvement step
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


            pi, log_pi, _, _, _ = self.sample((state_obs_batch, state_aug_batch, first_hidden_batch))

            qf1_pi, qf2_pi = self.critic((state_obs_batch, state_aug_batch, first_hidden_batch), pi)
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


            if updates % self.hyperps['critic_target_update'] == 0:
                soft_update(self.targ_critic1, self.critic1, tau)
                soft_update(self.targ_critic2, self.critic2, tau)


            #self.scaler.update()
            
            return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()











def run_sac(hyperps, device=torch.device("cuda"), save_dir='./nvme/', load_buffer_dir='./nvme/diskbuffer/'):

    
    print('Running SAC Simple, no bl no multiagent')

    os.environ['WANDB_MODE'] = 'run'
    wandb.init(config=hyperps, project="gpulab")


    mem_max_size = hyperps['maxmem']
    mem_start_thr = 0.01
    mem_train_thr = 0.0125

    memory = DiskBuffer(mem_max_size, filedir=load_buffer_dir)

    print('Batch size: {}'.format(hyperps['batch_size']))
    
    env = CarlaGymEnv.CarEnv(0, render=False, step_type="other", benchmark="Simple", auto_reset=False, discrete=False, sparse=False, dist_reward=True, display2d=False, distributed=False)


    sac_agent = SAC(env.action_space.shape, hyperps, device)


    wandb.watch(sac_agent.actor)
    wandb.watch(sac_agent.critic1)
    wandb.watch(sac_agent.critic2)

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

        obs = env.reset()
        
        old_obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}, max steps: {}, total_steps: {}, cumulative_reward: {}'.format(epi, hyperps['max_epochs'], hyperps['max_steps'], total_steps, cumulative_reward))
        
        cumulative_reward = 0

        for step_numb in range(hyperps['max_steps']):

            action, log_prob, mean, std, hidden = None, None, None, None, None
            sac_agent.eval()
            
            action, log_prob, mean, std, hidden = sac_agent.sample((old_obs[0], old_obs[1], old_hidden)) 

            
            q1, q2 = sac_agent.critic((old_obs[0], old_obs[1], old_hidden), action)

            obs, reward, done, info = env.step(action.cpu().detach().numpy()[0])

            reward = (w_vel*reward[0] + w_t*reward[1] + w_dis*reward[2] + w_col*reward[3] + w_lan*reward[4] + w_waypoint*reward[5])/6

            cumulative_reward += reward

            

            if(total_steps % 100 == 0):
              print('Final Sum Reward: {:.5f}'.format(reward))
              print('Total steps {}, max_steps: {}'.format(total_steps, hyperps['max_steps']))


            wandb.log({"final_r": reward, 'action':action.cpu(), 'log_prob':log_prob.cpu(), 'mean':mean.cpu(), 'std':std.cpu()})

            if(info != None):
                print(info)
           

            obs = (torch.Tensor(obs[0]).unsqueeze(0).transpose(1, 3).transpose(2, 3).float().to(device), torch.FloatTensor(obs[1]).to(device))

            memory.push((old_obs[0].to("cpu"), old_obs[1].to("cpu")), old_hidden.to("cpu"), action.to("cpu"), reward, hidden.to("cpu"), (obs[0].to("cpu"), obs[1].to("cpu")), done)

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


