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

import random

import glob

import torch.multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

from .arquitetures import SQNet, Actor, ActorSimple




class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, first_mem, action, reward, next_state, next_mem, done):
      experience = (state, first_mem, action, np.array([reward]), next_state, next_mem, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      first_mem_batch = []
      reward_batch = []
      next_state_batch = []
      next_mem_batch = []
      done_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, first_mem, action , reward, next_state, next_mem, done = experience
          state_batch.append(state)
          first_mem_batch.append(first_mem)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          next_mem_batch.append(next_mem)
          done_batch.append(done)

      #import pudb; pudb.set_trace()

      return (state_batch, first_mem_batch, action_batch, reward_batch, next_state_batch, next_mem_batch, done_batch)

  def __len__(self):
      return len(self.buffer)



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)






class ActorSimpleMem(nn.Module):

    def __init__(self, state_size, action_size, size_1=32, size_2=64, size_3=32, size_mem=256):
        super(ActorSimpleMem, self).__init__()

        self.state_size = state_size
        self.action_size = action_size


        self.fc1 = nn.Linear(state_size, size_1)

        self.fc2 = nn.Linear(size_1, size_2)

        self.fc3 = nn.Linear(size_2, size_3)

        #self.hidden = nn.Linear(size2, size3)

        self.hidden = nn.GRUCell(size_3, size_mem)

        self.mu = nn.Linear(size_mem, action_size)

        self.log_std = nn.Linear(size_mem, action_size)


    def forward(self, state_with_hidden):

        state, hidden = state_with_hidden

        x = state.reshape(-1, self.state_size)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        #x = F.relu(self.hidden(x))
        x = F.relu(self.fc3(x))

        #hidden.detach_()
        hidden = hidden.detach()

        hidden = self.hidden(x, hidden)

        #self.last_hidden = hidden

        mu = self.mu(F.relu(hidden))

        log_std = self.log_std(F.relu(hidden))

        return mu, log_std, hidden



class SAC():

  def __init__(self, env, obs_size, num_actions, hyperps, device, train=True):

    self.hyperps = hyperps
    self.env = env
    self.device = device

    self.num_actions = num_actions
    #import pudb; pudb.set_trace()
    if(len(obs_size) == 1):
        self.obs_state = obs_size[0]
        self.obs_state_size = obs_size[0]
        self.actor = ActorSimpleMem(self.obs_state, self.num_actions).to(device)
    else:
        self.obs_state = obs_size
        self.obs_state_size =  obs_size[0][0] * obs_size[0][1] * obs_size[1]
        self.actor = Actor(self.obs_state, self.num_actions).to(device)


    self.critic1 = SQNet(self.obs_state_size, self.num_actions).to(device)
    self.critic2 = SQNet(self.obs_state_size, self.num_actions).to(device)
    self.targ_critic1 = SQNet(self.obs_state_size, self.num_actions).to(device)
    self.targ_critic2 = SQNet(self.obs_state_size, self.num_actions).to(device)

    self.targ_critic1.load_state_dict(self.critic1.state_dict())
    self.targ_critic2.load_state_dict(self.critic2.state_dict())


    if(train):
        # initialize optimizers 
        self.q1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.hyperps['q_lr'])
        self.q2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.hyperps['q_lr'])
        self.policy_optimizer = optim.Adam(self.actor.parameters(), lr=self.hyperps['q_lr'])

        # entropy temperature
        self.alpha = self.hyperps['alpha']
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.hyperps['a_lr'])
    else:
        self.try_load()



  def sample(self, obs_with_hidden):

    mean, log_std, hidden = self.actor.forward(obs_with_hidden)

    log_std = torch.tanh(log_std)
    log_std = self.hyperps['log_std_min'] + 0.5 * (self.hyperps['log_std_max'] - self.hyperps['log_std_min']) * (log_std + 1)

    std = log_std.exp()
    normal = Normal(mean, std)
    
  
    x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    y_t = torch.tanh(x_t)

    action = y_t * self.hyperps['action_scale'] + self.hyperps['action_bias']
    log_prob = normal.log_prob(x_t)
    
    # Enforcing Action Bound
    
    log_prob = log_prob - torch.log(self.hyperps['action_scale'] * (1 - y_t.pow(2)) + self.hyperps['epsilon'])
    log_prob = log_prob.sum(1, keepdim=True)
    
    mean = torch.tanh(mean) * self.hyperps['action_scale'] + self.hyperps['action_bias']

    return action, log_prob, mean, hidden


  def try_load(self, save_dir='./project/savedmodels/rl/sac/'):
    print('Loading Model')
    paths = glob.glob(save_dir + '*_final.tar') ; step = 0
    #import pudb; pudb.set_trace()
    if len(paths) > 4:
        #ckpts = [int(s.split('.')[-2]) for s in paths]
        #ix = np.argmax(ckpts) ; step = ckpts[ix]
        #step = ckpts[ix]
        #import pudb; pudb.set_trace()
        self.targ_critic2.load_state_dict(torch.load(paths[-5]))
        self.targ_critic1.load_state_dict(torch.load(paths[-3]))
        self.critic2.load_state_dict(torch.load(paths[-4]))
        self.critic1.load_state_dict(torch.load(paths[-2]))
        self.actor.load_state_dict(torch.load(paths[-1]))

    #print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
    #return step


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


  def update(self, memory, updates):
        batch_size=self.hyperps['batch_size']
        alpha=self.hyperps['alpha']
        gamma=self.hyperps['gamma']
        tau=self.hyperps['tau']

        with torch.autograd.set_detect_anomaly(True):

            mem_state_batch, mem_first_mem_batch, mem_action_batch, mem_reward_batch, mem_next_state_batch, mem_next_mem_batch, mem_mask_batch = memory.sample(batch_size)
            #import pudb; pudb.set_trace()
            state_batch = torch.stack(mem_state_batch).to(self.device)
            first_hidden_batch = torch.stack(mem_first_mem_batch).to(self.device)
            next_state_batch = torch.stack(mem_next_state_batch).to(self.device)
            next_mem_batch = torch.stack(mem_next_mem_batch).to(self.device)
            action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
            reward_batch = torch.FloatTensor(np.stack(mem_reward_batch)).to(self.device)
            mask_batch = torch.FloatTensor(np.stack(mem_mask_batch)).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _, val_hidden = self.sample((next_state_batch, next_mem_batch.squeeze(1)))
                qf1_next_target, qf2_next_target = self.targ_critic1(next_state_batch, next_state_action), self.targ_critic2(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = reward_batch + (1 - mask_batch) * gamma * (min_qf_next_target)
            #
            qf1 = self.critic1(state_batch, action_batch)
            qf2 = self.critic2(state_batch, action_batch)
            # Two Q-functions to mitigate positive bias in the policy improvement step
                
            
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            pi, log_pi, _, pol_hidden = self.sample((state_batch, first_hidden_batch.squeeze(1)))
            
            qf1_pi = self.critic1(state_batch, pi)
            #import pudb; pudb.set_trace()
            qf2_pi = self.critic2(state_batch, pi)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            self.q1_optimizer.zero_grad()
            qf1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            qf2_loss.backward()
            self.q2_optimizer.step()


            pi, log_pi, _, last_hidden = self.sample((state_batch, first_hidden_batch.squeeze(1)))
            qf1_pi = self.critic1(state_batch, pi)
            qf2_pi = self.critic2(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()


            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs


        if updates % self.hyperps['critic_target_update'] == 0:
            soft_update(self.targ_critic1, self.critic1, tau)
            soft_update(self.targ_critic2, self.critic2, tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()



def run_sac(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True):

    hyperps['gru_steps'] = 8

    memory = BasicBuffer(hyperps['maxmem'])

    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    total_steps = 0
    updates = 0


    obs = env.reset()
    old_obs = obs

    log_reward = []

    epi = 0


    while(epi < hyperps['max_epochs']):

        for step_numb in range(hyperps['max_steps']):

            first_hidden = torch.zeros(1, 256).to(device)
            after_hidden = first_hidden.to(device)

            for gru_step in range(hyperps['gru_steps']):


                obs_hdd = (old_obs.unsqueeze(0).to(device), first_hidden.to(device))
                action, _, _, after_hidden = sac_agent.sample(obs_hdd)
                after_hidden.to(device) 
                action = np.asarray(action.cpu().detach()).T

                import pudb; pudb.set_trace()
                obs, reward, done, info = env.step(action)
                done |= total_steps == hyperps['max_steps']

                #import pudb; pudb.set_trace()

                memory.push(old_obs, first_hidden.to(device), action, reward, obs, after_hidden.to(device), done)

                old_obs = obs
                first_hidden = after_hidden

                total_steps += 1
                log_reward.append(reward)

                if done:
                    print('In SAC Done: {}'.format(len(memory)))
                    first_hidden = torch.zeros(1, 256).to(device)
                    obs = env.reset()
                    old_obs = obs                    
                    total_steps += 1
                    epi += 1 

                    #print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))

                    if(total_steps > hyperps['start_steps'] and epi % hyperps['log_interval'] == 0):
                        sac_agent.save_models(epi)
                        print('Saved sac, Average episode reward: {}'.format(sum(log_reward)/(len(log_reward) + 0.2)))

                    log_reward = []

 
                if render:
                    env.render()



            if len(memory) > hyperps['batch_size']*2 and total_steps % hyperps['update_every'] == 0:
                for i in range(hyperps['updates_per_step']):

                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(memory, updates)
                    updates += 1
                    if(epi % 2 == 0):
                        print('Updated Neural Nets. Losses: critic1:{:.4f}, critic2:{:.4f}, policy_loss:{:.4f}, entropy_loss: {:.4f}, alpha:{:.4f}.'.format(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
        

    
    sac_agent.save_models_final()
    return sac_agent






def demo_sac(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=False):

    sac_agent = SAC(env, obs_state, num_actions, hyperps, device, train=False)
    #import pudb; pudb.set_trace()
    total_steps = 0
    updates = 0

    for epi in range(hyperps['max_epochs']):
        obs = env.reset()
        
        old_obs = obs
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))

        log_reward = []

        for step_numb in range(hyperps['max_steps']):


            action, _, _ = sac_agent.sample(old_obs.unsqueeze(0).to(device)) 
            action = np.asarray(action.cpu().detach()).T
            if(epi % 3 == 0):
                print('Action: ' + str(action[0]), end=' ')
                print(str(action[1]))


            obs, reward, done, info = env.step(action)
            done |= total_steps == hyperps['max_steps']

            old_obs = obs

            total_steps += 1
            log_reward.append(reward)

            if done:
                print('In SAC Done')
                break


            if True:
                #print('Rendering')
                env.render()

    return sac_agent