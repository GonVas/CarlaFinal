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
import torch.multiprocessing as mp
from torch.multiprocessing import Lock
import os
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np

from .arquitetures import SQNet, Actor, ActorSimple


from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from torchviz import make_dot

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/sac_dist_exp1')

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



class SAC(nn.Module):

  def __init__(self, env, obs_size, num_actions, hyperps, device, train=True):
    super(SAC, self).__init__()

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


  def forward(self, obs_with_hidden):
    return self.sample(obs_with_hidden)

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


  def update(self, memory, updates, shared_model, shared_optimizer):
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
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40)
            for param, shared_param in zip(self.actor.parameters(), shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad # sync gradients with shared model
            #shared_optimizer.step()

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


class DistlSac(nn.Module):

    def __init__(self, len_rewards, env, obs_state, num_actions, hyperps, device, aggretator='min'):
        super(DistlSac, self).__init__()

        self.sacs = []
        for i in range(len_rewards):
            self.sacs.append(SAC(env, obs_state, num_actions, hyperps, device))

        func_maps = {'min':torch.min}

        aggretator = func_maps[aggretator]

        self.device = device
        self.hyperps = hyperps


    def sample(self, obs_with_hidden, single=False):

        sacs_out = []
        means = []
        log_stds = []

        obs, hidden = obs_with_hidden
        
        
        hiddens = hidden

        
        for i in range(len(self.sacs)):
            _, temp_mean, temp_log_std, temp_hidden = self.sacs[i].sample((obs, hiddens[i].unsqueeze(0)))
            means.append(temp_mean)
            log_stds.append(temp_log_std)
            hiddens[i] = temp_hidden


        
        mean = sum(means)/len(means)
        log_std = sum(log_stds)/len(log_stds)
        #hidden = torch.cat(tuple(hiddens), 1)

        #mean, log_std, hidden = self.actor.forward(obs_with_hidden)

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

        return action, log_prob, mean, hiddens

    def forward(self, obs_with_hidden):
        return self.sample(obs_with_hidden)

    def update(self, memory, updates, shared_model, shared_optimizer):
        batch_size=self.hyperps['batch_size']
        alpha=self.hyperps['alpha']
        gamma=self.hyperps['gamma']
        tau=self.hyperps['tau']


        mem_state_batch, mem_first_mem_batch, mem_action_batch, mem_reward_batch, mem_next_state_batch, mem_next_mem_batch, mem_mask_batch = memory.sample(batch_size)
       
        state_batch = torch.stack(mem_state_batch).to(self.device)
        
        first_hidden_batch = torch.stack(mem_first_mem_batch).to(self.device)
        #first_hidden_batch = mem_first_mem_batch
        next_state_batch = torch.stack(mem_next_state_batch).to(self.device)
        next_mem_batch = torch.stack(mem_next_mem_batch).to(self.device)
        action_batch = torch.FloatTensor(np.stack(mem_action_batch)).to(self.device)
        reward_batch = torch.FloatTensor(np.stack(mem_reward_batch)).to(self.device)
        mask_batch = torch.FloatTensor(np.stack(mem_mask_batch)).to(self.device).unsqueeze(1)

        qf1_losses = []
        qf2_losses = []
        policy_losses = []
        alpha_losses = []
        alpha_tlogs_losses = []

        for idx, sac in enumerate(self.sacs):

            curr_pol = sac.actor
            curr_crit1 = sac.critic1
            curr_crit2 = sac.critic2
            targ_critic1 = sac.targ_critic1
            targ_critic2 = sac.targ_critic2
            #torch.stack(first_hidden_batch[idx])

            first_mems = first_hidden_batch
            next_mems = next_mem_batch

            with torch.no_grad():
                next_state_action, next_state_log_pi, _, val_hidden = sac.sample((next_state_batch, next_mems[:, idx, :]))
                qf1_next_target, qf2_next_target = targ_critic1(next_state_batch, next_state_action), targ_critic2(next_state_batch, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = reward_batch[:, :, idx] + (1 - mask_batch) * gamma * (min_qf_next_target)
            

            qf1 = sac.critic1(state_batch, action_batch)
            qf2 = sac.critic2(state_batch, action_batch)
            # Two Q-functions to mitigate positive bias in the policy improvement step
            
            
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

            pi, log_pi, _, pol_hidden = sac.sample((state_batch, first_mems[:, idx, :]))
            
            qf1_pi = sac.critic1(state_batch, pi)
            #import pudb; pudb.set_trace()
            qf2_pi = sac.critic2(state_batch, pi)

            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            sac.q1_optimizer.zero_grad()
            qf1_loss.backward()
            sac.q1_optimizer.step()

            sac.q2_optimizer.zero_grad()
            qf2_loss.backward()
            sac.q2_optimizer.step()


            pi, log_pi, _, last_hidden = sac.sample((state_batch, first_mems[:, idx, :]))
            qf1_pi = sac.critic1(state_batch, pi)
            qf2_pi = sac.critic2(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((sac.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            
            sac.policy_optimizer.zero_grad()
            policy_loss.backward()

            """
            torch.nn.utils.clip_grad_norm_(sac.actor.parameters(), 40)
            for param, shared_param in zip(sac.actor.parameters(), shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad # sync gradients with shared model
            #shared_optimizer.step()
            """

            sac.policy_optimizer.step()


            alpha_loss = -(sac.log_alpha * (log_pi + sac.target_entropy).detach()).mean()

            sac.alpha_optim.zero_grad()
            alpha_loss.backward()
            sac.alpha_optim.step()

            sac.alpha = sac.log_alpha.exp()
            alpha_tlogs = sac.alpha.clone() # For TensorboardX logs


            if updates % sac.hyperps['critic_target_update'] == 0:
                soft_update(sac.targ_critic1, sac.critic1, tau)
                soft_update(sac.targ_critic2, sac.critic2, tau)


            qf1_losses.append(qf1_loss.item())
            qf2_losses.append(qf2_loss.item())
            policy_losses.append(policy_loss.item())
            alpha_losses.append(alpha_loss.item())
            alpha_tlogs_losses.append(alpha_tlogs.item())

        return qf1_losses, qf2_losses, policy_losses, alpha_losses, alpha_tlogs_losses




class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)



def train_on_env(env_name, rank, lock, shared_model, shared_optimizer, obs_state, num_actions, hyperps, shared_mem, device=torch.device("cpu"), render=True):

    hyperps['gru_steps'] = 8

    #memory = BasicBuffer(hyperps['maxmem'])

    memory = shared_mem

    env = env_name.fromrank(rank, withlock=True, distl=True, sparse=False)
    env._env.lock = lock

    #sac_agent0 = SAC(env, obs_state, num_actions, hyperps, device)
    #sac_agent1 = SAC(env, obs_state, num_actions, hyperps, device)
    #sac_agent2 = SAC(env, obs_state, num_actions, hyperps, device)
    #sac_agent3 = SAC(env, obs_state, num_actions, hyperps, device)
    #sac_agent4 = SAC(env, obs_state, num_actions, hyperps, device)


    distl_sac = DistlSac(5, env, obs_state, num_actions, hyperps, device)

    #sac_agent.policy_optimizer = shared_optimizer

    total_steps = 0
    updates = 0


    obs = env.reset()
    old_obs = obs

    log_reward = []

    epi = 0


    #writer.add_graph(distl_sac, [(torch.randn(4, 412), torch.randn(5, 4, 256))])
    #summary_model = ActorSimpleMem(412, 2)
    #summary_model.load_state_dict(shared_model.state_dict())
    #writer.add_graph(summary_model.cpu(), [(torch.randn(4, 412), torch.randn(4, 256))])
    

    #writer.close()
    #test_sample = (torch.randn(1, 1, 412).to(device), torch.randn(5, 256).to(device))
    #test_result = distl_sac(test_sample)

    #make_dot(test_result).render("attached", format="png")


    #writer.add_graph(DistlSac(5, env, obs_state, num_actions, hyperps, device), [(torch.randn(1, 1, 412).to(device), torch.randn(5, 256).to(device))])
    #writer.close()


    while(epi < hyperps['max_epochs']):

        #sac_agent.actor.load_state_dict(shared_model.state_dict()) # sync with shared model

        for step_numb in range(hyperps['max_steps']):

            first_hidden = torch.zeros(5, 256).to(device)
            after_hidden = first_hidden.to(device)

            for gru_step in range(hyperps['gru_steps']):

                
                obs_hdd = (old_obs.unsqueeze(0).to(device), first_hidden)
                action, _, _, after_hidden = distl_sac.sample(obs_hdd, single=True)
                #after_hidden.to(device) 
                action = np.asarray(action.cpu().detach()).T
                

                #make_dot(obs_hdd)

                #print('Rank: {} output action {}'.format(rank, str(action)))

                obs, reward, done, info = env.step(action)
                done |= total_steps == hyperps['max_steps']
                writer.add_scalar('Loss{}/reward0'.format(rank), reward[0], total_steps)
                writer.add_scalar('Loss{}/reward1'.format(rank), reward[1], total_steps)
                writer.add_scalar('Loss{}/reward2'.format(rank), reward[2], total_steps)
                writer.add_scalar('Loss{}/reward3'.format(rank), reward[3], total_steps)
                writer.add_scalar('Loss{}/reward3'.format(rank), reward[4], total_steps)
                #import pudb; pudb.set_trace()

                memory.push(old_obs, first_hidden, action, reward, obs, after_hidden, done)

                old_obs = obs
                first_hidden = after_hidden

                total_steps += 1
                log_reward.append(reward)

                if done:
                    print('Rank {}, Epoch:{}, MaxEpoch:{}, In SAC Done: {}'.format(rank, epi, hyperps['max_epochs'], len(memory)))
                    first_hidden = torch.zeros(5, 256).to(device)
                    obs = env.reset()
                    old_obs = obs                    
                    total_steps += 1
                    epi += 1 



                    if(total_steps > hyperps['start_steps'] and epi % hyperps['log_interval'] == 0):
                        distl_sac.save_models(epi)
                        print('Saved sac, Average episode reward: {}'.format(sum(log_reward)/(len(log_reward) + 0.2)))

                    log_reward = []

 
                if render:
                    env.render()


            if(epi > hyperps['max_epochs']):
                break


            if len(memory) > hyperps['batch_size']*2 and total_steps % hyperps['update_every'] == 0:
                for i in range(hyperps['updates_per_step']):

                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = distl_sac.update(memory, updates, shared_model, shared_optimizer)
                    writer.add_scalar('Loss{}/critic_1'.format(rank), sum(critic_1_loss), total_steps)
                    writer.add_scalar('Loss{}/critic_2'.format(rank), sum(critic_2_loss), total_steps)
                    writer.add_scalar('Loss{}/policy'.format(rank), sum(policy_loss), total_steps)
                    writer.add_scalar('Loss{}/ent'.format(rank), sum(ent_loss), total_steps)
                    writer.add_scalar('Loss{}/alpha'.format(rank), sum(alpha), total_steps)
                    print('Trained')
                    updates += 1
                    if(epi % 2 == 0):
                        pass
                        #print('Updated Neural Nets. Losses: critic1:{:.4f}, critic2:{:.4f}, policy_loss:{:.4f}, entropy_loss: {:.4f}, alpha:{:.4f}.'.format(critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha))


    
    #sac_agent.save_models_final()
    return sac_agent


def run_sac(env, obs_state, num_actions, hyperps, device=torch.device("cpu"), render=True):


    #import pudb; pudb.set_trace()
    #model = ActorSimpleMem(412, 2).share_memory().to(device)
    #shared_optimizer = SharedAdam(shared_model.parameters(), lr=hyperps['q_lr'])
   
    
   
    mem_buffer = BasicBuffer(hyperps['maxmem'])

    """
    for rank in range(3):
        p = mp.Process(target=train_on_env, args=(env, rank, lock, shared_model, shared_optimizer, obs_state, num_actions, hyperps, shared_buffer, device, render))
        time.sleep(10)
        p.start() ; processes.append(p)

    for p in processes: p.join()
    """
    train_on_env(env, 0, lock, shared_model, shared_optimizer, obs_state, num_actions, hyperps, shared_buffer, device, render)

    writer.close()
