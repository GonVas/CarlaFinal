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



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

from .arquitetures import SQNet, Actor, ActorSimple


from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchviz import make_dot

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sac_vanilla')


class BasicBuffer:

  def __init__(self, max_size):
      self.max_size = max_size
      self.buffer = deque(maxlen=max_size)

  def push(self, state, action, reward, next_state, done):
      experience = (state, action, np.array([reward]), next_state, done)
      self.buffer.append(experience)

  def sample(self, batch_size):
      state_batch = []
      action_batch = []
      reward_batch = []
      next_state_batch = []
      done_batch = []

      batch = random.sample(self.buffer, batch_size)

      for experience in batch:
          state, action, reward, next_state, done = experience
          state_batch.append(state)
          action_batch.append(action)
          reward_batch.append(reward)
          next_state_batch.append(next_state)
          done_batch.append(done)

      #import pudb; pudb.set_trace()

      return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

  def __len__(self):
      return len(self.buffer)



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)




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
        self.actor = ActorSimple(self.obs_state, self.num_actions).to(device)
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



  def sample(self, obs):

    mean, log_std = self.actor.forward(obs)

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

    return action, log_prob, mean


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

            mem_state_batch, mem_action_batch, mem_reward_batch, mem_next_state_batch, mem_mask_batch = memory.sample(batch_size)

            state_batch = torch.stack(mem_state_batch).to(self.device)
            next_state_batch = torch.stack(mem_next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
            reward_batch = torch.FloatTensor(np.stack(mem_reward_batch)).to(self.device)
            mask_batch = torch.FloatTensor(np.stack(mem_mask_batch)).to(self.device).unsqueeze(1)


            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.targ_critic1(next_state_batch, next_state_action), self.targ_critic2(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = reward_batch + (1 - mask_batch) * gamma * (min_qf_next_target)
            #
            qf1 = self.critic1(state_batch, action_batch)
            qf2 = self.critic2(state_batch, action_batch)
            # Two Q-functions to mitigate positive bias in the policy improvement step
                
            
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            pi, log_pi, _ = self.sample(state_batch)
            
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


            pi, log_pi, _ = self.sample(state_batch)
            qf1_pi = self.critic1(state_batch, pi)
            qf2_pi = self.critic2(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            import pudb; pudb.set_trace()
            #writer.add_graph(summary_model, [(torch.randn(64, 412), torch.randn(64, 256))])

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

    memory = BasicBuffer(hyperps['maxmem'])

    sac_agent = SAC(env, obs_state, num_actions, hyperps, device)

    total_steps = 0
    updates = 0



    #test_sample = torch.randn(1, 1, 412).to(device)
    
    #test_result = sac_agent(test_sample)

    

    for epi in range(hyperps['max_epochs']):
        obs = env.reset()
        print('Len of memory buffer: {}'.format(len(memory)))
        old_obs = obs
        #obs_t = torch.Tensor(obs).unsqueeze(0).transpose(1, 3).transpose(2, 3).float()
        
        total_steps += 1

        print('Epoch: {}, Max Epochs: {}'.format(epi, hyperps['max_epochs']))

        if(total_steps > hyperps['start_steps'] and epi % hyperps['log_interval'] == 0):
            sac_agent.save_models(epi)
            print('Saved sac, Average episode reward: {}'.format(sum(log_reward)/(len(log_reward) + 0.2)))
            log_reward = []

        log_reward = []

        for step_numb in range(hyperps['max_steps']):

            if(total_steps < hyperps['start_steps']):
                action = env.action_space.sample()
            else:
                action, _, _ = sac_agent.sample(old_obs.unsqueeze(0).to(device)) 
                action = np.asarray(action.cpu().detach()).T
                if(epi % 3 == 0):
                    print('Action: ' + str(action[0]), end=' ')
                    print(str(action[1]))


            obs, reward, done, info = env.step(action)
            done |= total_steps == hyperps['max_steps']

            #import pudb; pudb.set_trace()
            memory.push(old_obs, action, reward, obs, done)

            old_obs = obs

            total_steps += 1
            log_reward.append(reward)

            if done:
                print('In SAC Done: {}'.format(len(memory)))
                break

            #print('Len of memory: {}'.format(len(memory)))

            if len(memory) > hyperps['batch_size']*5 and total_steps % hyperps['update_every'] == 0:
                for i in range(hyperps['updates_per_step']):

                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = sac_agent.update(memory, updates)
                    updates += 1
                    if(epi % 2 == 0):
                        print('Updated Neural Nets. Losses: critic1:{:.4f}, critic2:{:.4f}, policy_loss:{:.4f}, entropy_loss: {:.4f}, alpha:{:.4f}.'.format(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))
            

            if render:
                env.render()

    
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